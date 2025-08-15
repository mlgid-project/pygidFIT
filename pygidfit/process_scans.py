import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Arc

import time


from pygidfit.fitting_models import (
    fit_clusters_multiprocessing,
    fit_ring_cluster,
    fit_peak_on_ring_cluster,
    fit_peak_cluster,
)

from pygidfit.box_utils import (
    boxes_preprocessing,
    make_box_attributes,
)

from pygidfit.io_utils import (
    DataLoader,
    DataSaver
)

from pygidfit.imgcontainer import ImageContainer
from pygidfit.clustering_and_errors import cluster_boxes_by_centers

def calc_smpl_hor(ai, crit_angle, wavelength):
    return np.pi * 4 / wavelength * np.sin(np.deg2rad((ai + crit_angle)) / 2) / 10


def img_preprocessing(img,  ai, crit_angle, wavelength, q_z):
    ## cut sample horiz
    crit_angle = np.radians(crit_angle)
    q_z_critical = calc_smpl_hor(ai, crit_angle, wavelength)
    mask = q_z < q_z_critical
    img[mask, :] = np.nan
    img[img <= 0] = np.nan
    return img


def _get_polar_grid(
        img_shape: Tuple[int, int],
        polar_shape: Tuple[int, int],
        beam_center: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    y0, z0 = beam_center

    y = (np.arange(img_shape[1]) - y0)
    z = (np.arange(img_shape[0]) - z0)

    yy, zz = np.meshgrid(y, z)
    rr = np.sqrt(yy ** 2 + zz ** 2)
    phi = np.arctan2(zz, yy)
    r_range = (rr.min(), rr.max())
    phi_range = phi.min(), phi.max()

    phi = np.linspace(*phi_range, polar_shape[0])
    r = np.linspace(*r_range, polar_shape[1])

    r_matrix = r[np.newaxis, :].repeat(polar_shape[0], axis=0)
    p_matrix = phi[:, np.newaxis].repeat(polar_shape[1], axis=1)

    polar_yy = r_matrix * np.cos(p_matrix) + y0
    polar_zz = r_matrix * np.sin(p_matrix) + z0

    return polar_yy, polar_zz, np.rad2deg(phi.max())

def polar_conversion(img: np.ndarray, yy: np.ndarray, zz: np.ndarray, algorithm: int) -> np.ndarray or None:
    return cv2.remap(img.astype(np.float32),
                        yy.astype(np.float32),
                        zz.astype(np.float32),
                        interpolation=algorithm,
                        borderValue=np.nan)




def fit_single_image(img, ai, crit_angle, wavelength,  q_xy, q_z, boxes,  yy, zz, clusters, debag = False,
                     multiprocessing = True):
    img = img_preprocessing(img, ai, crit_angle, wavelength, q_z)
    if debag:
        fig, axes = plt.subplots(figsize=(6, 6))
        norm = LogNorm(vmin=np.nanmin(img[img > 0]), vmax=np.nanmax(img))
        axes.imshow(img, extent=[q_xy.min(), q_xy.max(), q_z.min(), q_z.max()], cmap='viridis', origin='lower', norm=norm)

        for box in boxes:
            r_min, th1, r_max, th2 = map(float, box.boxes_q_deg)
            r = (r_min+r_max)/2

            arc = Arc((0, 0), width=2 * r, height=2 * r, angle=0,
                      theta1=th1, theta2=th2, color='red', linewidth=2)
            axes.add_patch(arc)
        plt.show()

    img = polar_conversion(img, yy, zz, cv2.INTER_LINEAR) ## polar image

    mask = np.zeros_like(img, dtype=bool)
    # for box in boxes:
    #     if not box.is_ring:
    #         xmin, ymin, xmax, ymax = map(int, box.limits)
    #         mask[ymin:ymax, xmin:xmax] = True
    for cluster in clusters:
        if cluster.type == 'peaks' or cluster.type == 'both':
            xmin, ymin, xmax, ymax = map(int, cluster.bbox)
            mask[ymin:ymax, xmin:xmax] = True

    # for cluster in clusters:
    #     for box, is_ring in zip(cluster.boxes, cluster.is_ring_list):
    #         if not is_ring:
    #             xmin, ymin, xmax, ymax = map(int, box)
    #             mask[ymin:ymax, xmin:xmax] = True

    # masked_img = np.where(mask, img, np.nan)
    masked_img = np.where(~mask, img, np.nan)

    if debag:
        fig, axes = plt.subplots(figsize=(6, 6))

        norm = LogNorm(vmin=np.nanmin(img[img > 0]), vmax=np.nanmax(img))
        axes.imshow(img, cmap='viridis', origin='lower', norm=norm)
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.limits)
            width = xmax - xmin
            height = ymax - ymin
            rect = Rectangle(
                (xmin, ymin), width, height,
                linewidth=1.5, edgecolor='red', facecolor='none'
            )
            axes.add_patch(rect)
        plt.show()

        fig, axes = plt.subplots(figsize=(6, 6))
        norm = LogNorm(vmin=np.nanmin(masked_img[masked_img > 0]), vmax=np.nanmax(masked_img))
        axes.imshow(masked_img, cmap='viridis', origin='lower', norm=norm)
        axes.title.set_text('masked image')
        plt.show()

        print("box example: ", boxes[0], "cluster example: ", clusters[0])
        # print("box example: ", boxes, "cluster example: ", clusters)
    time0 = time.time()

    if multiprocessing:
        fit_clusters_multiprocessing(clusters, boxes, img, masked_img, debag=debag)
    else:
        for cluster in clusters:
            if cluster.type == 'rings':
                fitting_result = fit_ring_cluster(cluster, boxes, masked_img, debag)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debag)
            elif cluster.type == 'peaks':
                fitting_result = fit_peak_cluster(cluster, boxes, img, debag)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debag)
        for cluster in clusters:
            if cluster.type == 'both':
                fitting_result = fit_peak_on_ring_cluster(cluster, boxes, img, debag)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debag)

            # elif cluster.type == 'both':
            #     cluster.fitting_result = fit_peak_on_ring_cluster(cluster, boxes, img, debag)
    time1 = time.time()
    print(f"image fitting took {(time1 - time0) * 1000} ms")



    # for i in range(len(clusters)):
    #     cluster = clusters[i]
    #     fit_cluster(cluster)





def fit_data(data, crit_angle,  yy, zz, debag, multiprocessing):

    ## boxes preprocessing
    data.boxes = []
    for i in range(len(data.detected_peaks)):
        data.boxes.append(boxes_preprocessing(data.detected_peaks[i],
                                              data.polar_shape, data.wavelength,
                                              data.q_abs_max))

    ## boxes clustering
    data.clusters = []
    for i in range(data.raw_giwaxs.shape[0]):
        data.clusters.append(cluster_boxes_by_centers(data.boxes[i]))
    ## image fitting
    img_container_list = []
    for i in range(data.raw_giwaxs.shape[0]):
        fit_single_image(data.raw_giwaxs[i], data.ai[i], crit_angle, data.wavelength,
                         data.q_xy, data.q_z, data.boxes[i],
                         yy, zz, data.clusters[i], debag, multiprocessing,
                         )
        img_container_list.append(_data2container(data.boxes[i], data.polar_shape, data.q_abs_max, data.ang_deg_max,
                                                  data.q_xy, data.q_z,
                                                  np.array(data.detected_peaks[i].visibility),
                                                  np.array(data.detected_peaks[i].score)))
    return img_container_list


def compute_qzqxy_with_error(radius, sigma_radius, angle_deg, sigma_angle_deg):
    """
    Computes qx and qy from radius and angle (in degrees) and propagates errors.

    Returns:
        q_array: np.array([qx, qy])
        sigma_array: np.array([sigma_qx, sigma_qy])
    """
    theta = np.deg2rad(angle_deg)
    sigma_theta = np.deg2rad(sigma_angle_deg)

    # coordinates
    qx = radius * np.sin(theta)
    qy = radius * np.cos(theta)

    # error propagation
    sigma_qx = np.sqrt((np.sin(theta) * sigma_radius) ** 2 + (radius * np.cos(theta) * sigma_theta) ** 2)
    sigma_qy = np.sqrt((np.cos(theta) * sigma_radius) ** 2 + (radius * np.sin(theta) * sigma_theta) ** 2)

    q_array = np.array([qx, qy])
    sigma_array = np.array([sigma_qx, sigma_qy])

    return q_array, sigma_array

def _data2container(boxes, polar_shape, q_abs_max, ang_deg_max, q_xy , q_z, visibility, score):

    img_container = ImageContainer()
    img_container.amplitude = np.array([box.fitting_result['amplitude'] for box in boxes])
    img_container.A = np.array([box.fitting_result['A'] for box in boxes]) * polar_shape[1] / q_abs_max
    img_container.B = np.array([box.fitting_result['B'] for box in boxes]) * polar_shape[0] / ang_deg_max
    img_container.C = np.array([box.fitting_result['C'] for box in boxes])
    img_container.theta = np.array([box.fitting_result['theta'] for box in boxes])
    img_container.radius = np.array([box.fitting_result['radius'] for box in boxes])/polar_shape[1]*q_abs_max
    img_container.radius_width = np.array([box.fitting_result['radius_width'] for box in boxes])/polar_shape[1]*q_abs_max
    angle_arr = np.array([box.fitting_result['angle'] for box in boxes]) / polar_shape[0] * ang_deg_max
    img_container.angle = np.nan_to_num(angle_arr, nan=45)
    img_container.angle_width = np.array([box.fitting_result['angle_width'] for box in boxes])/polar_shape[0]*ang_deg_max


    img_container.amplitude_err = np.array([box.fitting_error['amplitude'] for box in boxes])
    img_container.A_err  = np.array([box.fitting_error['A'] for box in boxes]) * polar_shape[1] / q_abs_max
    img_container.B_err  = np.array([box.fitting_error['B'] for box in boxes]) * polar_shape[0] / ang_deg_max
    img_container.C_err  = np.array([box.fitting_error['C'] for box in boxes])
    img_container.theta_err  = np.array([box.fitting_error['theta'] for box in boxes])
    img_container.radius_err  = np.array([box.fitting_error['radius'] for box in boxes])/polar_shape[1]*q_abs_max
    img_container.radius_width_err  = np.array([box.fitting_error['radius_width'] for box in boxes])/polar_shape[1]*q_abs_max
    img_container.angle_err  = np.array([box.fitting_error['angle'] for box in boxes])/polar_shape[0]*ang_deg_max
    img_container.angle_width_err  = np.array([box.fitting_error['angle_width'] for box in boxes])/polar_shape[0]*ang_deg_max

    img_container.qzqxyboxes, img_container.qzqxyboxes_err = compute_qzqxy_with_error(
        img_container.radius, img_container.radius_err, img_container.angle, img_container.angle_err)
    # img_container.qzqxyboxes = np.array([img_container.radius * np.sin(np.deg2rad(img_container.angle)),
    #                                      img_container.radius * np.cos(np.deg2rad(img_container.angle))])

    img_container.id = np.array([box.index for box in boxes])
    img_container.is_ring = np.array([box.is_ring for box in boxes])
    img_container.is_cut_qz = np.array([box.is_cut_qz for box in boxes])
    img_container.is_cut_qxy = np.array([box.is_cut_qxy for box in boxes])
    img_container.q_xy = q_xy
    img_container.q_z = q_z
    img_container.visibility = visibility
    img_container.score = score
    return img_container


def process_data_from_file(filename, batch_size = 10, crit_angle = 0, polar_shape = np.array([512,1024]), debag = False, multiprocessing = True):
    data_loaded = DataLoader(filename, batch_size=batch_size)
    entry_list = data_loaded.entry_list
    entry_done = data_loaded.entry_done
    batch_num = data_loaded.batch_num

    yy, zz, ang_deg_max = None, None, None

    if len(entry_list) != 0:
        for i in range(len(entry_list)):
            if debag:
                print("Current entry", entry_list[i])
            while not entry_done:
                data_loaded = DataLoader(filename=filename,
                                         entry_list=entry_list, entry_num=i, batch_num=batch_num,
                                         batch_size=batch_size,
                                         debag = debag
                                         )
                data = data_loaded.data
                data.polar_shape = polar_shape
                if yy is None or zz is None:
                    yy, zz, ang_deg_max = _get_polar_grid(data.raw_giwaxs.shape[1:], polar_shape, [0,0])
                data.ang_deg_max = ang_deg_max
                img_container_list = fit_data(data, crit_angle, yy, zz, debag, multiprocessing)
                DataSaver(img_container_list, filename, entry_list[i], batch_num, batch_size)
                entry_done = data_loaded.entry_done
                batch_num = data_loaded.batch_num + 1
            batch_num = 0
            entry_done = False
            return
    else:
        raise ValueError("No entries found in the HDF5 file.")

