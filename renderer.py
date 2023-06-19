import os
import trimesh
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

import envlight

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.1, far=10):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(np.radians(self.fovy) / 2)
        aspect = self.W / self.H
        return np.array([[1/(y*aspect),    0,            0,              0], 
                         [           0,  -1/y,            0,              0],
                         [           0,    0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)], 
                         [           0,    0,           -1,              0]], dtype=np.float32)

    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])

class GUI:
    def __init__(self, opt, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        
        self.mode = 'full'

        # init a sphere mesh
        mesh = trimesh.creation.icosphere()
        self.v = torch.from_numpy(mesh.vertices).float().contiguous().cuda()
        self.f = torch.from_numpy(mesh.faces).int().contiguous().cuda()
        self.vn = torch.from_numpy(mesh.vertex_normals).float().contiguous().cuda()
        self.roughness = 0
        self.metallic = 1

        # load light
        self.light = envlight.EnvLight(opt.hdr)
        self.FG_LUT = torch.from_numpy(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()
        
        if os.name == 'nt':
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        dpg.create_context()
        self.register_dpg()
        self.step()
        

    def __del__(self):
        dpg.destroy_context()
    
    def step(self):

        if self.need_update:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            # do MVP for vertices
            mv = torch.from_numpy(self.cam.view).cuda() # [4, 4]
            proj = torch.from_numpy(self.cam.perspective).cuda() # [4, 4]
            mvp = proj @ mv
            
            v_clip = torch.matmul(F.pad(self.v, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]

            rast, rast_db = dr.rasterize(self.glctx, v_clip, self.f, (self.H, self.W))
            
            if self.mode == 'depth':
                depth = rast[0, :, :, [2]]  # [H, W, 1]
                buffer = depth.detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
            else:
                # fake albedo (pure white)
                albedo = torch.tensor([1,1,1], dtype=torch.float32, device='cuda') * torch.ones(1, self.H, self.W, 3, dtype=torch.float32, device='cuda')
                albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
                if self.mode == 'albedo':
                    buffer = albedo[0].detach().cpu().numpy()
                else:
                    n, _ = dr.interpolate(self.vn.unsqueeze(0), rast, self.f)
                    n = safe_normalize(n)
                    if self.mode == 'normal':
                        buffer = (n[0].detach().cpu().numpy() + 1) / 2
                    else:
                        pos, _ = dr.interpolate(self.v.unsqueeze(0), rast, self.f)
                        campos = torch.linalg.inv(mv)[:3, 3]
                        v = safe_normalize(campos - pos)
                        NdotV = (n * v).sum(-1, keepdim=True)
                        l = NdotV * n * 2 - v
                        roughness = self.roughness * torch.ones(1, self.H, self.W, 1, dtype=torch.float32, device='cuda')
                        diffuse = self.light(n) * (1 - self.metallic) * albedo
                        if self.mode == 'diffuse':
                            buffer = diffuse[0].detach().cpu().numpy()
                        else:
                            fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1)
                            fg = dr.texture(self.FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, self.H, self.W, 2)
                            specular = self.light(l, roughness) * ((0.04 * (1 - self.metallic) + albedo * self.metallic) * fg[..., 0:1] + fg[..., 1:2])
                            specular = torch.where(rast[..., 3:] > 0, specular, torch.tensor(0).to(specular.device)) # remove background
                            if self.mode == 'specular':
                                buffer = specular[0].detach().cpu().numpy()
                            else: # 'full'
                                buffer = (diffuse + specular)[0].detach().cpu().numpy()

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if self.need_update:
                self.render_buffer = buffer
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + buffer) / (self.spp + 1)

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=200):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)              

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('albedo', 'depth', 'normal', 'diffuse', 'specular', 'full'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # metallic
                def callback_set_metallic(sender, app_data):
                    self.metallic = app_data
                    self.need_update = True

                dpg.add_slider_float(label="metallic", min_value=0, max_value=1.0, format="%.5f", default_value=self.metallic, callback=callback_set_metallic)
            
                # roughness
                def callback_set_roughness(sender, app_data):
                    self.roughness = app_data
                    self.need_update = True

                dpg.add_slider_float(label="roughness", min_value=0, max_value=1.0, format="%.5f", default_value=self.roughness, callback=callback_set_roughness)


            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='mesh viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr', default='/home/kiui/projects/envlight/assets/aerodynamics_workshop_2k.hdr', type=str)
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    opt = parser.parse_args()

    gui = GUI(opt)
    gui.render()