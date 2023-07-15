import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from Grasper.Grasp import Grasp
def result_process(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = ndimage.gaussian_filter(q_img, sigma=2.0, mode='nearest', truncate=2.0)
    ang_img = ndimage.gaussian_filter(ang_img, sigma=2.0, mode='nearest', truncate=2.0)
    width_img = ndimage.gaussian_filter(width_img, sigma=1.0, mode='nearest', truncate=2.0)

    return q_img, ang_img, width_img

def result_plot(fig, q_img, angle_img, width_img, depth_img):
    plt.ion()
    plt.clf()
    
    # Create imshow objects for each subplot
    ax_q = fig.add_subplot(1, 4, 1)
    ax_angle = fig.add_subplot(1, 4, 2)
    ax_width = fig.add_subplot(1, 4, 3)
    ax_depth = fig.add_subplot(1, 4, 4)
    
    # imshow and colorbar
    im_q = ax_q.imshow(q_img, cmap='jet', vmin=0, vmax=1)
    plt.colorbar(im_q)
    im_angle = ax_angle.imshow(angle_img, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
    plt.colorbar(im_angle)
    im_width = ax_width.imshow(width_img, cmap='jet', vmin=0, vmax=100)
    plt.colorbar(im_width)
    im_depth = ax_depth.imshow(np.squeeze(depth_img), cmap='gray')
    plt.colorbar(im_depth)
    
    # Set titles and turn off axes for all subplots
    ax_q.set_title('Q')
    ax_q.axis('off')
    ax_angle.set_title('Angle')
    ax_angle.axis('off')
    ax_width.set_title('Width')
    ax_width.axis('off')
    ax_depth.set_title('Depth')
    ax_depth.axis('off')
    
    # Redraw the canvas
    fig.canvas.draw()
    plt.pause(0.01)

def detect_grasps(q_img, ang_img, width_img=None, nu_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=nu_grasps)

    grasps = []
    for grasp_center_array in local_max:
        grasp_center = (grasp_center_array[1],grasp_center_array[0])

        grasp_angle = ang_img[grasp_center_array[0],grasp_center_array[1]]
        grasp_width = width_img[grasp_center_array[0],grasp_center_array[1]]
        g = Grasp(grasp_center,grasp_angle,grasp_width)
        grasps.append(g)

    return grasps