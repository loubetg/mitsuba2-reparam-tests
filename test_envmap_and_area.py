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

test_name = "test_envmap_and_area"

def make_scene(integrator, spp, param):
    return load_string("""
        <?xml version="1.0"?>
        <scene version="2.0.0">

            {integrator}

            <sensor type="perspective">
                <string name="fov_axis" value="smaller"/>
                <float name="near_clip" value="0.1"/>
                <float name="far_clip" value="2800"/>
                <float name="focus_distance" value="1000"/>
                <transform name="to_world">
                    <lookat origin="0, 0, 10" target="0, 0, 0" up="0, 1, 0"/>
                </transform>
                <float name="fov" value="10"/>
                <sampler type="independent">
                    <integer name="sample_count" value="{spp}"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="250"/>
                    <integer name="height" value="250"/>
                    <rfilter type="box"/>
                </film>
            </sensor>

            <emitter type="envmap">
                <float name="scale" value="1"/>
                <string name="filename" value="data/maps/sun.exr"/>
                <transform name="to_world">
                    <rotate x="1.0" angle="90"/>
                </transform>
            </emitter>

            <shape type="obj" id="light_shape">
                <transform name="to_world">
                    <rotate x="1" angle="180"/>
                    <translate x="10.0" y="0.0" z="15.0"/>
                </transform>
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <emitter type="smootharea" id="smooth_area_light">
                    <spectrum name="radiance" value="100"/>
                </emitter>
            </shape>

            <shape type="obj" id="object">
                <string name="filename" value="data/meshes/smooth_empty_cube.obj"/>
                <bsdf type="diffuse" id="objectmat">
                </bsdf>
                <transform name="to_world">
                    <translate z="0.6"/>
                    <translate x="{param}"/>
                </transform>
            </shape>

            <shape type="obj" id="planemesh">
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="0.8 0.8 0.8"/>
                </bsdf>
                <transform name="to_world">
                    <scale value="2.0"/>
                </transform>
            </shape>
        </scene>
    """.format(integrator=integrator, spp=spp, param=param))

def get_diff_param(scene):

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

# Test settings and integrators

fd_eps = 0.001
fd_spp = 256
fd_passes = 10
fd_integrator = """<integrator type="path">
                       <integer name="max_depth" value="2"/>
                   </integrator>"""

diff_spp = 4
diff_passes = 10
diff_integrator = """<integrator type="pathreparam">
                         <integer name="max_depth" value="2"/>
                         <boolean name="use_convolution_envmap" value="false"/>
                     </integrator>"""

test_finite_difference(test_name, make_scene, get_diff_param,
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, fd_eps)
