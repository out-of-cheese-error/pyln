# pyln

[comment]: <> ([![Build status]&#40;https://github.com/out-of-cheese-error/pyln/workflows/build/badge.svg?branch=master&event=push&#41;]&#40;https://github.com/out-of-cheese-error/pyln/actions?query=workflow%3Abuild&#41;)
[comment]: <> ([![Python Version]&#40;https://img.shields.io/pypi/pyversions/pyln.svg&#41;]&#40;https://pypi.org/project/pyln/&#41;)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/out-of-cheese-error/pyln/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/out-of-cheese-error/pyln/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/out-of-cheese-error/pyln/releases)
[![License](https://img.shields.io/github/license/out-of-cheese-error/pyln)](https://github.com/out-of-cheese-error/pyln/blob/master/LICENSE)

## a (WIP) Python 3D line art engine
This is a complete Python port of [fogleman/ln](https://github.com/fogleman/ln) (with some help from [ln.py](https://github.com/ksons/ln.py)) using NumPy, Numba, and Pillow.

### Examples
Images rendered from the scripts in the [examples folder](examples):

#### [Cube](examples/cube.py)
![cube](examples/images/cube.svg)

#### [Cubes](examples/cubes.py)
![cubes](examples/images/cubes.svg)

#### [Function](examples/function.py)
![function](examples/images/function.svg)

[comment]: <> (#### [Beads]&#40;examples/beads.py&#41;)

[comment]: <> (![beads]&#40;examples/images/beads.svg&#41;)

#### [CSG (Constructive Solid Geometry)](examples/csg.py)
![csg](examples/images/csg.gif)

#### [Outline](examples/outline.py)
![outline](examples/images/outline.svg)

[comment]: <> (#### [Cones]&#40;examples/cones.py&#41;)

[comment]: <> (![cones]&#40;examples/images/cones.svg&#41;)

#### [Skyscrapers](examples/skyscrapers.py)
![skyscrapers](examples/images/skyscrapers.svg)

#### [Striped Skyscrapers](examples/skyscrapers_striped.py)
![skyscrapers_striped](examples/images/skyscrapers_striped.svg)

#### [Suzanne (from .obj file)](examples/suzanne.py)
![suzanne](examples/images/suzanne.svg)

#### [Suzanne (voxelized, from .stl file)](examples/voxelize.py)
![voxelize](examples/images/voxelize.svg)

## Credits

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template).
