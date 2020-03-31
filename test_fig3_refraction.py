import numpy as np
import os
import mitsuba
import enoki as ek

mts_variant = 'rgb'
mitsuba.set_variant('gpu_autodiff_' + mts_variant)

from mitsuba.core import Transform4f, Bitmap
from mitsuba.core.xml import load_string
from mitsuba.python.util import traverse
from utils import test_finite_difference

test_name = "test_fig3_refraction"

def make_scene(integrator, spp, translateX, rotate, scale):
    return load_string("""
        <?xml version="1.0"?>
        <scene version="2.0.0">

            {integrator}

            <sensor type="perspective">
                <string name="fov_axis" value="smaller"/>
                <float name="near_clip" value="0.1"/>
                <float name="far_clip" value="2800"/>
                <float name="focus_distance" value="10"/>
                <transform name="to_world">
                    <lookat origin="0, 10, 6" target="0, 0, 1.5" up="0, 0, 1"/>
                </transform>
                <float name="fov" value="25.00"/>
                <sampler type="independent">
                    <integer name="sample_count" value="{spp}"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="128"/>
                    <integer name="height" value="128"/>
                    <rfilter type="box"/>
                </film>
            </sensor>

            <shape type="obj">
                <transform name="to_world">
                    <rotate x="1" angle="180"/>
                    <translate x="0" y="0" z="100"/>
                    <rotate x="1.0" y="0.0" z="0.0" angle="45"/>
                    <rotate x="0.0" y="1.0" z="0.0" angle="0"/>                
                </transform>
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <emitter type="smootharea" id="smooth_area_light">
                    <spectrum name="radiance" value="6000"/>
                </emitter>
            </shape>

            <shape type="obj">
                <string name="filename" value="data/meshes/slab_random.obj"/>
                <transform name="to_world">
                    <scale value="3.5"/>
                    <rotate x="0.0" y="0.0" z="1.0" angle="0"/>
                    <translate x="0.0" y="1.7" z="0.0"/>
                </transform>
                <bsdf type="roughdielectric">
                    <float name="alpha" value="0.01"/>
                </bsdf>
            </shape>

            <shape type="obj" id="object">
                <string name="filename" value="data/meshes/empty_cube.obj"/>
                <transform name="to_world">
                    <scale value="2.0"/>
                    <rotate x="0.0" y="0.0" z="1.0" angle="45.0"/>
                    <translate x="0.0" y="-1.0" z="1.5"/>
                    <rotate x="0.0" y="10.0" z="5.5" angle="{rotate}"/>
                    <translate x="{translateX}" y="0.0" z="0.0"/>
                    <scale value="{scale}"/>
                </transform>
                <bsdf type="diffuse">
                    <spectrum name="reflectance" value="0.5"/>
                </bsdf>
            </shape>

            <shape type="obj" id="planemesh">
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <bsdf type="diffuse">
                    <texture type="checkerboard" name="reflectance">
                        <transform name="to_uv">
                            <scale value="10.0"/>
                        </transform>
                        <spectrum name="color0" value="0.6"/>
                        <spectrum name="color1" value="0.2"/>
                    </texture>
                </bsdf>
                <transform name="to_world">
                    <scale value="20.0"/>
                    <translate x="0.0" y="0.0" z="0.0"/>
                    <rotate x="1.0" y="0.0" z="0.0" angle="0"/>
                </transform>
            </shape>
        </scene>
    """.format(integrator=integrator, spp=spp,
               translateX=translateX, rotate=rotate, scale=scale))

# Rotation (axis (0, 10, 5.5))

def make_scene_rotation(integrator, spp, param):
    return make_scene(integrator, spp, 0, param, 1)

def get_diff_param_rotation(scene):

    # Create a differentiable hyperparameter
    diff_param = mitsuba.core.Float(0.0);
    ek.set_requires_gradient(diff_param);

    # Update vertices so that they depend on diff_param
    properties = traverse(scene)
    t = mitsuba.core.Transform4f.rotate(mitsuba.core.Vector3f(0, 10,5.5),  diff_param)
    vertex_positions = properties['object.vertex_positions']
    vertex_positions_t = t.transform_point(vertex_positions)
    properties['object.vertex_positions'] = vertex_positions_t

    # Update the scene
    properties.update()

    return diff_param

# Translation (X axis)

def make_scene_translation(integrator, spp, param):
    return make_scene(integrator, spp, param, 0, 1)

def get_diff_param_translation(scene):

    # Create a differentiable hyperparameter
    diff_param = mitsuba.core.Float(0.0);
    ek.set_requires_gradient(diff_param);

    # Update vertices so that they depend on diff_param
    properties = traverse(scene)
    t = mitsuba.core.Transform4f.translate(mitsuba.core.Vector3f(1.0,0.0,0.0) * diff_param)
    vertex_positions = properties['object.vertex_positions']
    vertex_positions_t = t.transform_point(vertex_positions)
    properties['object.vertex_positions'] = vertex_positions_t

    # Update the scene
    properties.update()

    return diff_param

# Scale

def make_scene_scale(integrator, spp, param):
    return make_scene(integrator, spp, 0, 0, param+1)

def get_diff_param_scale(scene):

    # Create a differentiable hyperparameter
    diff_param = mitsuba.core.Float(1.0);
    ek.set_requires_gradient(diff_param);

    # Update vertices so that they depend on diff_param
    properties = traverse(scene)
    vertex_positions = properties['object.vertex_positions']
    vertex_positions_t = vertex_positions * diff_param
    properties['object.vertex_positions'] = vertex_positions_t

    # Update the scene
    properties.update()

    return diff_param

fd_spp = 256
fd_passes = 10
fd_integrator = """<integrator type="path">
                       <integer name="max_depth" value="4"/>
                   </integrator>"""

diff_spp = 16
diff_passes = 10
diff_integrator = """<integrator type="pathreparam">
                         <integer name="max_depth" value="4"/>
                     </integrator>"""

test_finite_difference(test_name + "/rotation", make_scene_rotation, get_diff_param_rotation,
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, 0.02)

test_finite_difference(test_name + "/translation", make_scene_translation, get_diff_param_translation,
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, 0.002)

test_finite_difference(test_name + "/scale", make_scene_scale, get_diff_param_scale,
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, 0.005)