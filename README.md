# mitsuba2-reparam-tests

This repository contains a few tests for the reparameterization technique
described in the article [Reparameterizing discontinuous integrands for
differentiable rendering](https://rgl.epfl.ch/publications/Loubet2019Reparameterizing)
and implemented with [Mitsuba 2](https://github.com/loubetg/mitsuba2/tree/reparam).

## Compiling and running the tests

To run the tests, make sure that you have compiled the version of Mistuba 2
that contains the plugin `src/integrators/path-reparam.cpp`:

    git clone --branch reparam --recursive https://github.com/loubetg/mitsuba2

This branch has a default configuration that enables the compilation of GPU
modes. Such modes are needed since all the tests rely on automatic
differentiation. Mitsuba 2 can be compiled on Linux using:

    cmake -GNinja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DMTS_ENABLE_OPTIX=ON .
    ninja

The CMake options `-DMTS_OPTIX_PATH` and `-DCMAKE_CUDA_COMPILER` can be used to specify the location of
Cuda and OptiX 6.5 on the system. Please see the
[Mitsuba 2 documentation](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/compiling/)
for details about the compilation. Once Mitsuba 2 is compiled, the command

    source setpath.sh

must be run in the Mitsuba directory. Then, the tests can be executed with Python, using
the version of Python that has been used to compile Mitsuba 2. For instance:

    python3 test_light_position.py

## Partial derivatives tests

These tests render a scene and generate images with partial derivatives with
respect to particular parameters of the scene, computed using the
reparameterization technique. Similar images are also computed using finite
differences for comparison.

Misc. tests:
- `test_light_position.py`: derivatives wrt the position of the light source.
- `test_light_position_pair.py`: derivatives wrt the position of two light sources.
- `test_object_position.py`: derivatives wrt the position of a mesh.
- `test_envmap.py`: derivatives in shadows with a sun envmap.
- `test_envmap_and_area.py`: derivatives with both an envmap and a smooth area light.

From Fig. 3 in the paper:
- `test_fig3_direct_visiblity.py`: derivatives on silhouettes.
- `test_fig3_shadows.py`: derivatives in shadows.
- `test_fig3_glossy_reflection.py`: derivatives in reflected light.
- `test_fig3_refraction.py`: derivatives in refraction.

From Fig. 6 in the paper:
- `test_fig6_variance.py`: shows the effect of variance reduction and noise in diffuse scattering.

From Fig. 11 in the paper:
- `test_fig11_corkscrew.py`: test for bias at silhouettes.
- `test_fig11_bias.py`: test for bias in shadows.


## Optimization tests

These tests perform some optimizations of scene parameters from target images,
based on gradient descent.

- `optim_colors.py`: optimization of a rgb texture.
- `optim_pose.py`: optimization of the position of an object.
- `optim_vertices.py`: optimization of the vertices of an object.
- `optim_light_position.py`: optimization of the position of light source.
