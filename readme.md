# Envlight

A toolkit to load `*.hdr` environment light map and sample lighting under the split-sum approximation.

This is basically a wrapped version of [`light.py` from nvdiffrec](https://github.com/NVlabs/nvdiffrec/blob/main/render/light.py).

### Install

Assume `torch` and [`nvdiffrast`](https://nvlabs.github.io/nvdiffrast/#linux) already installed.

```bash
pip install git+https://github.com/ashawkey/envlight

# or locally
git clone https://github.com/ashawkey/envlight
cd envlight
pip install .
```

### Usage

```python
import envlight

normal # [..., 3], assume normalized, in [-1, 1]
reflective # [..., 3], assume normalized, in [-1, 1]
roughness # [..., 1], in [0, 1]

light = envlight.EnvLight('envlight/assets/aerodynamics_workshop_2k.hdr', device='cuda')

diffuse = light(normal) # [..., 3]
specular = light(reflective, roughness) # [..., 3]

```

An example renderer:
```bash
# requries extra dependencies: pip install trimesh dearpygui
python renderer.py
```

https://github.com/ashawkey/envlight/assets/25863658/70974921-d7bc-4189-980f-11663e2e127b


### Acknowledgement
* Credits to Nvidia's [nvdiffrec](https://github.com/NVlabs/nvdiffrec).
