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

test_name = "test_corkscrew"

def make_scene(integrator, spp, param):
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
                    <lookat origin="2, 0, 0" target="0, 0, 0" up="0, 1, 0"/>
                </transform>
                <float name="fov" value="10.0"/>
                <sampler type="independent">
                    <integer name="sample_count" value="{spp}"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="190"/>
                    <integer name="height" value="80"/>
                    <rfilter type="box"/>
                    <string name="pixel_format" value="rgb"/>
                </film>
            </sensor>

            <shape type="obj" id="smooth_area_light_shape">
                <transform name="to_world">
                    <scale value="3.0"/>
                    <rotate x="1" angle="180"/>
                    <translate x="0.0" y="0.0" z="50.0"/>
                    <rotate y="1" angle="90"/>
                </transform>
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <emitter type="area" id="smooth_area_light">
                    <spectrum name="radiance" value="300"/>
                </emitter>
            </shape>

            <shape type="obj" id="objectmesh">
                <string name="filename" value="data/meshes/corkscrew.obj"/>
                <transform name="to_world">
                    <scale value="1.0"/>
                    <rotate x="0.0" y="0.0" z="1.0" angle="{param}"/>
                </transform>
                <bsdf type="diffuse">
                    <spectrum name="reflectance" value="0.3"/>
                </bsdf>
            </shape>
           
        </scene>
    """.format(integrator=integrator, spp=spp, param=param))

def get_diff_param(scene):

    # Create a differentiable hyperparameter
    diff_param = mitsuba.core.Float(0.0);
    ek.set_requires_gradient(diff_param);

    # Update vertices so that they depend on diff_param
    properties = traverse(scene)
    t = mitsuba.core.Transform4f.rotate(mitsuba.core.Vector3f(0.0,0.0,1.0), diff_param)
    vertex_positions = properties['objectmesh.vertex_positions']
    vertex_positions_t = t.transform_point(vertex_positions)
    properties['objectmesh.vertex_positions'] = vertex_positions_t

    # Update the scene
    properties.update()

    return diff_param

# Test settings and integrators

fd_eps = 1
fd_spp = 512
fd_passes = 1
fd_integrator = """<integrator type="path">
                       <integer name="max_depth" value="2"/>
                   </integrator>"""

diff_spp = 32
diff_passes = 10
diff_integrator = """<integrator type="pathreparam">
                         <integer name="max_depth" value="2"/>
                     </integrator>"""

test_finite_difference(test_name, make_scene, get_diff_param, 
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, fd_eps)
