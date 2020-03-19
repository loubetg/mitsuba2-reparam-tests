import numpy as np
import os
import mitsuba
import enoki as ek

mts_variant = 'rgb'
mitsuba.set_variant('gpu_autodiff_' + mts_variant)

from mitsuba.core import Transform4f, Bitmap, Float, Vector3f
from mitsuba.core.xml import load_string
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, SGD

# This test optimizes the vertex positions of an object.

path = "output/optim_vertices/"

def make_scene(integrator, spp):
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
                <float name="fov" value="37.3077"/>
                <sampler type="independent">
                    <integer name="sample_count" value="{spp}"/>
                </sampler>
                <film type="hdrfilm">
                    <integer name="width" value="250"/>
                    <integer name="height" value="200"/>
                    <rfilter type="box" >
                        <float name="radius" value="0.5"/>
                    </rfilter>
                </film>
            </sensor>

            <shape type="obj" id="smooth_area_light_shape">
                <transform name="to_world">
                    <rotate x="1" angle="180"/>
                    <translate x="10.0" y="0.0" z="15.0"/>
                </transform>
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <emitter type="smootharea" id="smooth_area_light">
                    <spectrum name="radiance" value="100"/>
                </emitter>
            </shape>

            <shape type="obj" id="planemesh">
                <string name="filename" value="data/meshes/xy_plane.obj"/>
                <bsdf type="roughplastic">
                    <float name="alpha" value="0.01"/>
                </bsdf>
                <transform name="to_world">
                    <scale value="2.0"/>
                </transform>    
            </shape>
        </scene>
    """.format(integrator=integrator, spp=spp))

# Define integrators for this test

path_str =  """<integrator type="path">
                   <integer name="max_depth" value="2"/>
               </integrator>"""

path_reparam_str =  """<integrator type="pathreparam">
                           <integer name="max_depth" value="2"/>
                           <boolean name="use_variance_reduction" value="true"/>
                           <boolean name="use_convolution" value="true"/>
                           <boolean name="disable_gradient_diffuse" value="true"/>
                       </integrator>"""

if not os.path.isdir(path):
    os.makedirs(path)

# Render the target image

scene = make_scene(path_str, 4);
fsize = scene.sensors()[0].film().size()
image_ref = render(scene)
write_bitmap(path + "out_ref.exr", image_ref, fsize)
print("Writing " + path + "out_ref.exr")

# Define the differentiable scene for the optimization

del scene
scene = make_scene(path_reparam_str, 8);

properties = traverse(scene)

key = 'planemesh.vertex_positions'

print("Target positions:")
print(properties[key])

properties.keep([key])

# Add an offset from target positions
properties[key] += 0.3
properties.update()

print("Initial positions:")
print(properties[key])

# Instantiate an optimizer
opt = SGD(properties, lr=15.0, momentum=0.9)

for i in range(100):

    image = render(scene)

    image_np = image.numpy().reshape(fsize[1], fsize[0], 3)
    output_file = path + 'out_%03i.exr' % i
    print("Writing image %s" % (output_file))
    Bitmap(image_np).write(output_file)

    # Objective function
    loss = ek.hsum(ek.hsum(ek.sqr(image - image_ref))) / (fsize[1]*fsize[0]*3)
    print("Iteration %i: loss=%f" % (i, loss[0]))

    ek.backward(loss)
    opt.step()
