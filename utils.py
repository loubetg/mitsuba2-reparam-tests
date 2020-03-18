import numpy as np
import os
from mitsuba.core import Bitmap
from mitsuba.python.autodiff import render
from enoki import *

# Convert signed floats to blue/red gradient image
def write_gradient_image(grad, name, fsize):
    
    # print("grad min", grad.min())
    # print("grad max", grad.max())

    # Compute RGB channels for .exr image (no grad = black)

    grad_R = grad.copy()
    grad_R[grad_R<0] = 0.0
    grad_B = grad.copy()
    grad_B[grad_B>0] = 0.0
    grad_B *= -1.0
    grad_G = grad.copy()*0.0;

    grad_np = np.concatenate((grad_R,grad_G,grad_B), axis=2)

    print('Writing', name+ ".exr")
    Bitmap(grad_np).write(name+ ".exr")

    # Compute RGB channels for .png image (no grad = white)

    grad_clamped = grad.copy()
    grad_clamped *= 3.0  # Arbitrary, for visualization
    grad_clamped[grad_clamped > 1] = 1
    grad_clamped[grad_clamped < -1] = -1
    grad_R = grad_clamped.copy()
    grad_G = grad_clamped.copy()
    grad_B = grad_clamped.copy()

    pos = grad_clamped >= 0
    neg = grad_clamped < 0
    grad_R[neg] = (1.0 + grad_clamped)[neg]
    grad_R[pos] = 1.0
    grad_G[neg] = (1.0 + grad_clamped)[neg]
    grad_G[pos] = (1.0 - grad_clamped)[pos]
    grad_B[neg] = 1.0
    grad_B[pos] = (1.0 - grad_clamped)[pos]
    grad_np = np.concatenate((grad_R,grad_G,grad_B), axis=2)

    print('Writing', name + ".png")
    Bitmap(((np.clip(grad_np, a_min = 0.0, a_max = 1.0))*255).astype(np.uint8)).write(name + ".png")

def render_gradient(scene, spp, pass_count, scale, path, P):

    sensor = scene.sensors()[0]
    fsize = sensor.film().size()

    for i in range(pass_count):
        y_i = render(scene)
        forward(P, i == pass_count - 1)

        nb_channels = len(y_i) // (fsize[1] * fsize[0])
        grad_i = gradient(y_i).numpy().reshape(fsize[1], fsize[0], nb_channels)
        grad_i = grad_i[:, :, 0]

        grad_i[grad_i != grad_i] = 0
        if i == 0:
            y = y_i.numpy()
            y[y != y] = 0
            grad = grad_i
        else:
            temp = detach(y_i).numpy()
            temp[temp != temp] = 0
            y += temp
            del y_i
            grad += grad_i

    grad /= pass_count
    y /= pass_count

    if (scale == 0.0):
        scale = np.abs(grad).max()

    grad = grad.reshape(fsize[1], fsize[0], 1)

    write_gradient_image(grad / scale, path + 'gradient', fsize)

    y_np = y.reshape(fsize[1], fsize[0], nb_channels)
    print('Writing ' + path + 'radiance.exr')
    Bitmap(y_np).write(path + 'radiance.exr')
    
    return grad

def test_finite_difference(test_name, make_scene, get_diff_param, 
                           diff_integrator, diff_spp, diff_passes,
                           fd_integrator, fd_spp, fd_passes, fd_eps):

    print("Running test:", test_name)

    path = "output/" + test_name + "/"
    if not os.path.isdir(path):
        os.makedirs(path)

    print("Rendering finite differences...")

    scene_fd0 = make_scene(fd_integrator, fd_spp, -fd_eps)
    scene_fd1 = make_scene(fd_integrator, fd_spp,  fd_eps)
    fsize = scene_fd0.sensors()[0].film().size()

    for i in range(fd_passes):
        if i == 0:
            values_fd0 = render(scene_fd0).numpy()
            values_fd1 = render(scene_fd1).numpy()
        else:
            values_fd0 += render(scene_fd0).numpy()
            values_fd1 += render(scene_fd1).numpy()

    values_fd0 /= fd_passes
    values_fd1 /= fd_passes

    channels = len(values_fd0) // fsize[0] // fsize[1]

    values_fd0.resize(fsize[1], fsize[0], channels)
    values_fd1.resize(fsize[1], fsize[0], channels)

    print("Writing " + path + 'radiance_fd0.exr')
    Bitmap(values_fd0).write(path + 'radiance_fd0.exr')
    print("Writing " + path + 'radiance_fd1.exr')
    Bitmap(values_fd1).write(path + 'radiance_fd1.exr')

    gradient_fd = (values_fd1-values_fd0)/(2.0*fd_eps);
    gradient_fd = gradient_fd[:,:,[0]]

    scale = np.abs(gradient_fd).max()
    write_gradient_image(gradient_fd/scale, path + 'gradient_fd', fsize)

    del scene_fd0, scene_fd1, values_fd0, values_fd1, channels, gradient_fd

    print("Rendering gradients... ({} spp, {} passes)".format(diff_spp, diff_passes))

    scene = make_scene(diff_integrator, diff_spp, 0.0);
    assert scene is not None
    diff_param = get_diff_param(scene)
    gradient_rp_np = render_gradient(scene, diff_spp, diff_passes, scale, path, diff_param)