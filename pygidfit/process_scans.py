import numpy as np
import cv2
from typing import Tuple, Any

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for lib in ["matplotlib" , "numba", "h5py"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
import time
from dataclasses import dataclass
import pygid


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
    save_fit,
    DetectedPeaks,
    read_detected_peaks,
)

from pygidfit.imgcontainer import ImageContainer
from pygidfit.clustering_and_errors import cluster_boxes_by_centers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

def calc_smpl_hor(ai, crit_angle, wavelength):
    return np.pi * 4 / wavelength * np.sin(np.deg2rad((ai + crit_angle)) / 2) / 10


def img_preprocessing(img,  ai, crit_angle, wavelength, q_z):
    ## cut sample horiz
    # crit_angle = np.radians(crit_angle)
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


def fit_single_image(polar_img, boxes, clusters, peaks_pool = None, debug = False,
                     multiprocessing = False):

    # if debug:
    #     fig, axes = plt.subplots(figsize=(6, 6))
    #     norm = LogNorm(vmin=np.nanmin(img[img > 0]), vmax=np.nanmax(img))
    #     axes.imshow(img, extent=[q_xy.min(), q_xy.max(), q_z.min(), q_z.max()], cmap='inferno', origin='lower', norm=norm)
    #     for box in boxes:
    #         r_min, th1, r_max, th2 = map(float, box.boxes_q_deg)
    #         r = (r_min+r_max)/2
    #
    #         arc = Arc((0, 0), width=2 * r, height=2 * r, angle=0,
    #                   theta1=th1, theta2=th2, color='red', linewidth=2)
    #         axes.add_patch(arc)
    #     plt.show()


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

    time1 = time.time()
    if debug:
        print(f"image fitting took {(time1 - time0) * 1000} ms")




def compute_qzqxy_with_error(radius, sigma_radius, angle_deg, sigma_angle_deg, qz_max, qxy_max):
    """
    Computes qx and qy from radius and angle (in degrees) and propagates errors.

    Returns:
        q_array: np.array([qx, qy])
        sigma_array: np.array([sigma_qx, sigma_qy])
    """
    theta = np.deg2rad(angle_deg)
    sigma_theta = np.deg2rad(sigma_angle_deg)

    # coordinates
    qz = radius * np.sin(theta)
    qxy = radius * np.cos(theta)

    # error propagation
    sigma_qz = np.sqrt((np.sin(theta) * sigma_radius) ** 2 + (radius * np.cos(theta) * sigma_theta) ** 2)
    sigma_qxy = np.sqrt((np.cos(theta) * sigma_radius) ** 2 + (radius * np.sin(theta) * sigma_theta) ** 2)
    sigma_array = np.array([sigma_qz, sigma_qxy])

    qz = np.minimum(qz, qz_max)
    qxy = np.sqrt(radius ** 2 - qz ** 2)
    qxy = np.minimum(qxy, qxy_max)
    qz = np.sqrt(radius ** 2 - qxy ** 2)

    angle_deg = np.rad2deg(np.arctan2(qz, qxy))

    q_array = np.array([qz, qxy])
    return q_array, sigma_array, radius, angle_deg

def _data2container(boxes, polar_shape, q_abs_max, ang_deg_max, q_xy_max , q_z_max, wavelength):

    img_container = ImageContainer()
    img_container.amplitude = np.array([box.fitting_result['amplitude'] for box in boxes])
    img_container.A = np.array([box.fitting_result['A'] for box in boxes]) * polar_shape[1] / q_abs_max
    img_container.B = np.array([box.fitting_result['B'] for box in boxes]) * polar_shape[0] / ang_deg_max
    img_container.C = np.array([box.fitting_result['C'] for box in boxes])
    img_container.theta = np.array([box.fitting_result['theta'] for box in boxes])

    radius_arr = np.array([box.fitting_result['radius'] for box in boxes])/polar_shape[1] * q_abs_max
    angle_arr = np.array([box.fitting_result['angle'] for box in boxes]) / polar_shape[0] * ang_deg_max
    angle_arr = np.nan_to_num(angle_arr, nan=45)
    angle_arr[angle_arr < 2] = 0


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

    # scaling and *2 as in mlgidGUI and mlgidDETECT
    img_container.radius_width = np.array([box.fitting_result['radius_width'] for box in boxes])/polar_shape[1]*q_abs_max*2
    img_container.angle_width = np.array([box.fitting_result['angle_width'] for box in boxes])/polar_shape[0]*ang_deg_max*2

    img_container.amplitude_err = np.array([box.fitting_error['amplitude'] for box in boxes])
    img_container.A_err = np.array([box.fitting_error['A'] for box in boxes]) * polar_shape[1] / q_abs_max
    img_container.B_err = np.array([box.fitting_error['B'] for box in boxes]) * polar_shape[0] / ang_deg_max
    img_container.C_err = np.array([box.fitting_error['C'] for box in boxes])
    img_container.theta_err = np.array([box.fitting_error['theta'] for box in boxes])
    img_container.radius_err = np.array([box.fitting_error['radius'] for box in boxes])/polar_shape[1]*q_abs_max
    img_container.radius_width_err = np.array([box.fitting_error['radius_width'] for box in boxes])/polar_shape[1]*q_abs_max
    img_container.angle_err = np.array([box.fitting_error['angle'] for box in boxes])/polar_shape[0]*ang_deg_max
    img_container.angle_width_err = np.array([box.fitting_error['angle_width'] for box in boxes])/polar_shape[0]*ang_deg_max

    img_container.qzqxyboxes, img_container.qzqxyboxes_err, img_container.radius, img_container.angle = compute_qzqxy_with_error(
        img_container.radius, img_container.radius_err, img_container.angle, img_container.angle_err,
        q_z_max, q_xy_max)
    # img_container.qzqxyboxes = np.array([img_container.radius * np.sin(np.deg2rad(img_container.angle)),
    #                                      img_container.radius * np.cos(np.deg2rad(img_container.angle))])

    img_container.id = np.array([box.index for box in boxes])
    img_container.is_ring = np.array([box.is_ring for box in boxes])
    img_container.is_cut_qz = np.array([box.is_cut_qz for box in boxes])
    img_container.is_cut_qxy = np.array([box.is_cut_qxy for box in boxes])
    # img_container.q_xy = q_xy
    # img_container.q_z = q_z
    # img_container.visibility = visibility
    # img_container.score = score
    return img_container


@dataclass
class ProcessDataFromFile:
    filename: str
    entry: str | None = None
    frame_num: int | None = None
    crit_angle: float = 0.0
    polar_shape: Any = None
    ratio_threshold: float = 50.0
    clustering_distance_peaks: float = 10.0
    clustering_distance_rings: float = 10.0
    clustering_extend: float = 2.0
    use_pool: bool = False
    debug: bool = False
    multiprocessing: bool = False

    def __post_init__(self):
        if self.polar_shape is None:
            self.polar_shape = [512, 1024]
        self.process_data_from_file()

    def process_data_from_file(self):
        self.nexus = pygid.NexusFile(self.filename)
        self.entry_dict = self.nexus.entry_dict
        if self.entry is None:
            for entry in self.entry_dict:
                self.process_single_entry(entry)
            return
        if not self.entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self.process_single_entry(self.entry)

    def process_single_entry(self, entry):
        self.yy = None
        self.zz = None
        self.ang_deg_max = None
        self.peaks_pool = [] if self.use_pool else None

        frame_num_all = self.entry_dict[entry]['shape'][0]

        if self.frame_num is None:
            for frame_num in range(frame_num_all):
                self.process_single_frame(entry, frame_num)
            return
        if self.frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        self.process_single_frame(entry, self.frame_num)


    def process_single_frame(self, entry, frame_num):
        # load conversion
        conversion = self.nexus.load_entry(entry, frame_num)
        q_xy = conversion.matrix[0].q_xy
        q_z = conversion.matrix[0].q_z
        img_reciprocal = conversion.img_gid_q[0]
        ai = conversion.params.ai[0]
        wavelength = conversion.params.wavelength

        # image preprocessing
        img_reciprocal = img_preprocessing(img_reciprocal, ai, self.crit_angle, wavelength, q_z)

        if self.yy is None or self.zz is None or self.ang_deg_max is None:
            self.yy, self.zz, self.ang_deg_max = (
                _get_polar_grid(img_reciprocal.shape, self.polar_shape, [0, 0]))
        polar_img = polar_conversion(img_reciprocal, self.yy, self.zz, cv2.INTER_LINEAR)

        # load detected peaks
        detected_peaks = read_detected_peaks(self.nexus, entry, frame_num)

        # run fitting, update pool
        img_container_fit, self.peaks_pool = fit_data(polar_img, detected_peaks['radius'],  detected_peaks['radius_width'],
                                     detected_peaks['angle'],  detected_peaks['angle_width'],
                                     wavelength, np.nanmax(q_xy), np.nanmax(q_z), self.ang_deg_max,
                                     self.ratio_threshold, self.clustering_distance_peaks,
                                     self.clustering_distance_rings, self.clustering_extend,
                                     self.debug, self.multiprocessing, self.peaks_pool)

        img_container_fit.score = detected_peaks['score']
        img_container_fit.visibility = detected_peaks['visibility']
        img_container_fit.q_xy = q_xy
        img_container_fit.q_z = q_z

        # save img_container_fit
        save_fit(self.filename, entry, img_container_fit, frame_num)


def fit_data(polar_img, radius, radius_width, angle, angle_width, wavelength, q_xy_max, q_z_max, ang_deg_max = 90,
             ratio_threshold = 50, clustering_distance_peaks = 10,
             clustering_distance_rings = 10, clustering_extend = 2, debug = False, multiprocessing = False, peaks_pool = None):
    polar_shape = polar_img.shape
    # boxes preprocessing
    detected_peaks = DetectedPeaks(radius = radius, radius_width = radius_width, angle = angle, angle_width = angle_width)
    boxes = boxes_preprocessing(detected_peaks,
                        polar_shape, wavelength,
                        np.sqrt(q_xy_max**2 + q_z_max**2), ratio_threshold,
                        q_xy_max, q_z_max)
    # clustarization
    clusters = cluster_boxes_by_centers(boxes, clustering_distance_peaks, clustering_distance_rings, clustering_extend)

    # real calling of fitting
    fit_single_image(polar_img, boxes, clusters, peaks_pool=peaks_pool, debug=debug,
                     multiprocessing=multiprocessing)

    img_container = _data2container(boxes, polar_shape, np.sqrt(q_xy_max**2 + q_z_max**2), ang_deg_max,
                                    q_xy_max, q_z_max,
                                    wavelength)
    if peaks_pool is None:
        return img_container, None
    else:
        return img_container, boxes

