# mitsuba2-reparam-tests

This repository contains a few tests for the reparameterization technique described in the article [Reparameterizing discontinuous integrands for differentiable rendering](https://rgl.epfl.ch/publications/Loubet2019Reparameterizing) and implemented with Mitsuba 2. 

## Compiling and running the tests

To run the tests, make sure that you have compiled the version of Mistuba 2 that contains the plugin `src/integrators/path-reparam.cpp` (https://github.com/loubetg/mitsuba2). Mitsuba 2 must be compiled with GPU modes enabled because all these tests rely on automatic differentiation. Please see the [Mitsuba 2 documentation](https://mitsuba2.readthedocs.io/en/latest/) for details.

Once Mitsuba 2 is compiled, you need to run `source setpath.sh` (Unix) in the Mitsuba directory. Then, the tests can be executed with python, using the version of python that has been used to compile Mitsuba 2. For instance: `python3.7 test_light_position.py`.

## Partial derivatives tests

- `test_light_position.py`: test derivatives wrt the position of the light source. 
- `test_object_position.py`: test derivatives wrt the position of a mesh.
- `test_corkscrew.py`: test similar to Fig. 11 (a) in the paper. 
- `test_bias.py`: test similar to Fig. 11 (b) in the paper.

## Optimization tests

- `optim_colors.py`: optimization of a rgb texture.
- `optim_pose.py`: optimization of the position of an object.
- `optim_vertices.py`: optimization of the vertices of an object.
- `optim_light_position.py`: optimization of the position of light source.
