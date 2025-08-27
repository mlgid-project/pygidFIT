import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Arc
import logging
for lib in ["matplotlib" , "numba", "h5py"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
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
    DataSaver,
    DetectedPeaks,
    DataBatch
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def show_masked_images_debug(img, masked_img, boxes, clusters, debug=True):
    """
    Display images with boxes and print example data if debug is True.

    Parameters
    ----------
    img : np.ndarray
        Original image.
    masked_img : np.ndarray
        Masked version of the image.
    boxes : list
        List of box objects with a 'limits' attribute (xmin, ymin, xmax, ymax).
    clusters : list
        List of cluster objects.
    debug : bool
        If True, display plots and print examples.
    """
    if not debug:
        return

    # Show original image with boxes
    fig, axes = plt.subplots(figsize=(6, 6))
    norm = LogNorm(vmin=np.nanmin(img[img > 0]), vmax=np.nanmax(img))
    axes.imshow(img, cmap='inferno', origin='lower', norm=norm)

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

    # Show masked image
    fig, axes = plt.subplots(figsize=(6, 6))
    norm = LogNorm(vmin=np.nanmin(masked_img[masked_img > 0]), vmax=np.nanmax(masked_img))
    axes.imshow(masked_img, cmap='inferno', origin='lower', norm=norm)
    axes.set_title('masked image')
    plt.show()

    # Print example data
    print("box example: ", boxes[0], "cluster example: ", clusters[0])
    # print("box all: ", boxes, "cluster all: ", clusters)


def fit_single_image(img, ai, crit_angle, wavelength,  q_xy, q_z, boxes,  yy, zz, clusters, peaks_pool = None, debug = False,
                     multiprocessing = False, polar_img = None):
    if polar_img is None:
        img = img_preprocessing(img, ai, crit_angle, wavelength, q_z)
        polar_img = polar_conversion(img, yy, zz, cv2.INTER_LINEAR)
        # import fabio
        # edf_file = fabio.edfimage.EdfImage(data=polar_img.astype(np.float32))
        # edf_file.write(r"D:\PhD\mlgid\pygidFIT\polar_image.edf")


    if debug:
        fig, axes = plt.subplots(figsize=(6, 6))
        norm = LogNorm(vmin=np.nanmin(img[img > 0]), vmax=np.nanmax(img))
        axes.imshow(img, extent=[q_xy.min(), q_xy.max(), q_z.min(), q_z.max()], cmap='inferno', origin='lower', norm=norm)
        for box in boxes:
            r_min, th1, r_max, th2 = map(float, box.boxes_q_deg)
            r = (r_min+r_max)/2

            arc = Arc((0, 0), width=2 * r, height=2 * r, angle=0,
                      theta1=th1, theta2=th2, color='red', linewidth=2)
            axes.add_patch(arc)
        plt.show()

    ## polar image

    mask = np.zeros_like(polar_img, dtype=bool)

    for cluster in clusters:
        if cluster.type == 'peaks' or cluster.type == 'both':
            xmin, ymin, xmax, ymax = map(int, cluster.bbox)
            mask[ymin:ymax, xmin:xmax] = True

    masked_img = np.where(~mask, polar_img, np.nan)

    show_masked_images_debug(polar_img, masked_img, boxes, clusters, debug=debug)

    time0 = time.time()

    if multiprocessing:
        fit_clusters_multiprocessing(clusters, boxes, polar_img, masked_img, debug=debug)
    else:
        for cluster in clusters:
            if cluster.type == 'rings':
                fitting_result = fit_ring_cluster(cluster, boxes, masked_img, peaks_pool, debug)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)
            elif cluster.type == 'peaks':
                fitting_result = fit_peak_cluster(cluster, boxes, polar_img, peaks_pool, debug)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)
        for cluster in clusters:
            if cluster.type == 'both':
                fitting_result = fit_peak_on_ring_cluster(cluster, boxes, polar_img, peaks_pool, debug)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)


    if peaks_pool is not None:
        peaks_pool = boxes


    time1 = time.time()
    if debug:
        print(f"image fitting took {(time1 - time0) * 1000} ms")
    return peaks_pool



def fit_data(data, crit_angle,  yy, zz, peaks_pool, ratio_threshold,
        clustering_distance,  clustering_extend, debug, multiprocessing):

    ## boxes preprocessing
    data.boxes = []
    for i in range(len(data.detected_peaks)):
        data.boxes.append(boxes_preprocessing(data.detected_peaks[i],
                                              data.polar_shape, data.wavelength,
                                              data.q_abs_max, ratio_threshold,
                                              np.max(data.q_xy), np.max(data.q_z)))

    ## boxes clustering
    data.clusters = []
    for i in range(data.raw_giwaxs.shape[0]):
        data.clusters.append(cluster_boxes_by_centers(data.boxes[i], clustering_distance, clustering_extend))
    ## image fitting
    img_container_list = []
    for i in range(data.raw_giwaxs.shape[0]):
        peaks_poll = fit_single_image(data.raw_giwaxs[i], data.ai[i], crit_angle, data.wavelength,
                         data.q_xy, data.q_z, data.boxes[i],
                         yy, zz, data.clusters[i], peaks_pool, debug, multiprocessing,
                         )
        img_container_list.append(_data2container(data.boxes[i], data.polar_shape, data.q_abs_max, data.ang_deg_max,
                                                  data.q_xy, data.q_z,
                                                  np.array(data.detected_peaks[i].visibility),
                                                  np.array(data.detected_peaks[i].score), data.wavelength))
    return img_container_list, peaks_poll


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

def _data2container(boxes, polar_shape, q_abs_max, ang_deg_max, q_xy , q_z, visibility, score, wavelength):

    img_container = ImageContainer()
    img_container.amplitude = np.array([box.fitting_result['amplitude'] for box in boxes])
    img_container.A = np.array([box.fitting_result['A'] for box in boxes]) * polar_shape[1] / q_abs_max
    img_container.B = np.array([box.fitting_result['B'] for box in boxes]) * polar_shape[0] / ang_deg_max
    img_container.C = np.array([box.fitting_result['C'] for box in boxes])
    img_container.theta = np.array([box.fitting_result['theta'] for box in boxes])

    radius_arr = np.array([box.fitting_result['radius'] for box in boxes])/polar_shape[1]*q_abs_max
    angle_arr = np.array([box.fitting_result['angle'] for box in boxes]) / polar_shape[0] * ang_deg_max
    angle_arr = np.nan_to_num(angle_arr, nan=45)


    def adjust_missing_wedge(wavelength, q_abs, phi):
        k = 2 * np.pi / float(wavelength)

        q_abs = np.asarray(q_abs)
        phi = np.asarray(phi)

        mask = np.abs(q_abs) > np.abs(2 * k * np.cos(np.deg2rad(phi)))
        phi_corrected = phi.copy()

        phi_corrected[mask] = np.rad2deg(
            np.arccos(np.clip(q_abs[mask] / (2 * k), -1.0, 1.0))
        )

        return phi_corrected

    angle_arr = adjust_missing_wedge(wavelength, radius_arr, angle_arr)
    img_container.radius = radius_arr
    img_container.angle = angle_arr

    img_container.radius_width = np.array([box.fitting_result['radius_width'] for box in boxes])/polar_shape[1]*q_abs_max

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


def process_data_from_file(filename, batch_size = 10, crit_angle = 0, polar_shape = np.array([512,1024]),
                           ratio_threshold = 50, clustering_distance = 7, clustering_extend = 2,
                           use_poll = False, debug = False, multiprocessing = False):
    data_loaded = DataLoader(filename, batch_size=batch_size)
    entry_list = data_loaded.entry_list
    entry_done = data_loaded.entry_done
    batch_num = data_loaded.batch_num

    yy, zz, ang_deg_max = None, None, None

    peaks_poll = [] if use_poll else None

    if len(entry_list) != 0:
        for i in range(len(entry_list)):
            if debug:
                print("Current entry", entry_list[i])
            while not entry_done:
                data_loaded = DataLoader(filename=filename,
                                         entry_list=entry_list, entry_num=i, batch_num=batch_num,
                                         batch_size=batch_size,
                                         debug = debug
                                         )
                data = data_loaded.data
                data.polar_shape = polar_shape
                if yy is None or zz is None:
                    yy, zz, ang_deg_max = _get_polar_grid(data.raw_giwaxs.shape[1:], polar_shape, [0,0])
                    # height, width = data.raw_giwaxs.shape[1:]
                    # top = np.column_stack((np.zeros(height), np.arange(height)))
                    # right = np.column_stack((np.arange(width), np.full(width, height - 1)))
                    # bottom = np.column_stack((np.full(height, width - 1), np.arange(height - 1, -1, -1)))
                    # left = np.column_stack((np.arange(width - 1, -1, -1), np.zeros(width)))
                    # frame_points = np.vstack([top, right, bottom, left])
                    # frame_y = frame_points[:, 1]  # old image rows
                    # frame_x = frame_points[:, 0]  # old image columns
                    # tolerance = 1.0
                    # # Flatten yy and zz
                    # yy_flat = yy.ravel()  # shape (N_pixels,)
                    # zz_flat = zz.ravel()
                    # indices = np.arange(yy_flat.size)
                    #
                    # # Compute distance squared to all frame points at once
                    # # (yy_flat[:, None] - frame_y[None, :])**2 + (zz_flat[:, None] - frame_x[None, :])**2 <= tolerance**2
                    # mask_flat = np.any(
                    #     (np.abs(yy_flat[:, None] - frame_y[None, :]) <= tolerance) &
                    #     (np.abs(zz_flat[:, None] - frame_x[None, :]) <= tolerance),
                    #     axis=1
                    # )
                    #
                    # # Convert back to 2D coordinates
                    # new_y, new_x = np.unravel_index(indices[mask_flat], yy.shape)
                    #
                    # plt.figure(figsize=(8, 8))
                    # plt.scatter(new_x, new_y, s=1, c='b')
                    # plt.gca().invert_yaxis()
                    # plt.axis('equal')
                    # plt.title("Mapped frame of the original image (fast)")
                    # plt.show()


                data.ang_deg_max = ang_deg_max
                img_container_list, peaks_poll = fit_data(data, crit_angle, yy, zz, peaks_poll,
                                                          ratio_threshold,
                                                          clustering_distance,
                                                          clustering_extend,
                                                          debug, multiprocessing)
                DataSaver(img_container_list, filename, entry_list[i], batch_num, batch_size)
                entry_done = data_loaded.entry_done
                batch_num = data_loaded.batch_num + 1
            batch_num = 0
            entry_done = False
    else:
        raise ValueError("No entries found in the HDF5 file.")




def process_data_img_container(img_container, crit_angle = 0, polar_shape = np.array([512,1024]),
                            ratio_threshold = 50, clustering_distance = 7, clustering_extend = 2,
                           use_poll = False, debug = False, multiprocessing = False):
    detected_peaks = DetectedPeaks()
    data = DataBatch()
    detected_peaks.radius = img_container.radius
    detected_peaks.radius_width = np.abs(img_container.radius_width)
    detected_peaks.angle = img_container.angle
    detected_peaks.angle_width = np.abs(img_container.angle_width)
    detected_peaks.score = img_container.scores
    data.raw_reciprocal = img_container.raw_reciprocal
    data.converted_polar_image = img_container.converted_polar_image[0][0]
    data.polar_shape = img_container.converted_polar_image.shape[-2:]
    data.ai = img_container.ai
    data.wavelength = img_container.wavelength
    data.q_abs_max = np.sqrt(img_container.q_xy**2 + img_container.q_z**2)
    data.ang_deg_max = 90
    data.q_xy = np.linspace(0, img_container.q_xy, img_container.raw_reciprocal.shape[1])
    data.q_z = np.linspace(0, img_container.q_z, img_container.raw_reciprocal.shape[0])

    data.boxes = boxes_preprocessing(detected_peaks,
                                              data.polar_shape, data.wavelength,
                                              data.q_abs_max, ratio_threshold,
                                                np.max(data.q_xy), np.max(data.q_z))
    data.clusters = cluster_boxes_by_centers(data.boxes, clustering_distance, clustering_extend)
    peaks_pool = None
    yy, zz = None, None


    peaks_poll = fit_single_image(data.raw_reciprocal, data.ai, crit_angle, data.wavelength,
                                  data.q_xy, data.q_z, data.boxes,
                                  yy, zz, data.clusters, peaks_pool, debug, multiprocessing,
                                  data.converted_polar_image
                                  )
    img_container = _data2container(data.boxes, data.polar_shape, data.q_abs_max, data.ang_deg_max,
                                                  data.q_xy, data.q_z,
                                                  np.array([0]*len(detected_peaks.score)),
                                                  np.array(detected_peaks.score),
                                                  data.wavelength)

    return img_container
