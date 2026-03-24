import numpy as np
import cv2
from typing import Tuple, Any, Optional
import importlib.metadata
import logging
from datetime import datetime

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
    """
    Calculate the critical scattering vector (q) for the sample horizon.

    Parameters
    ----------
    ai : float
        Incident angle of the beam (degrees).
    crit_angle : float
        Critical angle of the sample (degrees).
    wavelength : float
        Wavelength of the radiation (meters).

    Returns
    -------
    float
        Critical q value corresponding to the sample horizon (Å⁻¹).
    """
    return np.pi * 4 / wavelength * np.sin(np.deg2rad((ai + crit_angle)) / 2) / 10


def img_preprocessing(img,  ai, crit_angle, wavelength, q_z):
    """
    Preprocess a reciprocal-space image by masking below the critical q_z
    and non-positive pixels.

    Parameters
    ----------
    img : np.ndarray
        2D reciprocal-space image to preprocess.
    ai : float
        Incident angle of the beam (degrees).
    crit_angle : float
        Critical angle of the sample (degrees).
    wavelength : float
        Radiation wavelength (meters).
    q_z : np.ndarray
        Out-of-plane scattering vector array corresponding to the image.

    Returns
    -------
    np.ndarray
        Preprocessed image with masked pixels replaced by NaN.
    """
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
    """
    Generate a polar coordinate grid for a 2D image.

    Parameters
    ----------
    img_shape : tuple of int
        Shape of the original 2D image (rows, columns).
    polar_shape : tuple of int
        Shape of the polar grid to generate (polar angles, radial bins).
    beam_center : tuple of float
        Coordinates of the beam center (y0, z0) in the image.

    Returns
    -------
    polar_yy : np.ndarray
        Y-coordinates of the polar grid in Cartesian space.
    polar_zz : np.ndarray
        Z-coordinates of the polar grid in Cartesian space.
    phi_max_deg : float
        Maximum polar angle in degrees.
    """
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
    """
    Convert a 2D image to polar coordinates using a remapping grid.

    Parameters
    ----------
    img : np.ndarray
        Original 2D image to convert.
    yy : np.ndarray
        X-coordinate grid for remapping (from `_get_polar_grid`).
    zz : np.ndarray
        Y-coordinate grid for remapping (from `_get_polar_grid`).
    algorithm : int
        Interpolation method (e.g., cv2.INTER_LINEAR, cv2.INTER_NEAREST).

    Returns
    -------
    np.ndarray
        Polar-transformed 2D image.
    """

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


def fit_single_image(polar_img, boxes, clusters, theta_fixed=True, peaks_pool=None, debug=False,
                     multiprocessing=False):
    """
    Fit clusters in a polar-transformed image with Gaussian models.

    This function applies Gaussian fitting to clusters of peaks, rings, or combined
    peak-on-ring clusters. It optionally masks irrelevant areas of the image.

    Parameters
    ----------
    polar_img : ndarray
        2D polar-transformed scattering image (axis 0: polar angle, axis 1: radial coordinate |q|).
    boxes : list
        List of `Boxes` objects containing detected peak information.
    clusters : list
        List of cluster objects, each representing peaks, rings, or both.
    theta_fixed : bool, optional
        Fix Gaussian tilt angle to 0° (azimuthal direction) during fitting. Default is True.
    peaks_pool : list or None, optional
        Pool of previously fitted peaks to reuse parameters for sequential frames. Default is None.
    debug : bool, optional
        Enable debug mode to visualize masked images and fitting results. Default is False.
    multiprocessing : bool, optional
        Enable multiprocessing for cluster fitting. Default is False.

    Notes
    -----
    - Clusters of type 'peaks' and 'both' are masked before fitting.
    - Depending on cluster type, the appropriate fitting function is called:
        - 'peaks' → `fit_peak_cluster`
        - 'rings' → `fit_ring_cluster`
        - 'both' → `fit_peak_on_ring_cluster`
    - Fitted parameters are stored back into the `boxes` objects.
    - If debug mode is enabled, execution time is printed.
    """
    mask = np.zeros_like(polar_img, dtype=bool)

    for cluster in clusters:
        if cluster.type == 'peaks' or cluster.type == 'both':
            xmin, ymin, xmax, ymax = map(int, cluster.bbox)
            mask[ymin:ymax, xmin:xmax] = True

    masked_img = np.where(~mask, polar_img, np.nan)

    show_masked_images_debug(polar_img, masked_img, boxes, clusters, debug=debug)

    time0 = time.time()

    if multiprocessing:
        fit_clusters_multiprocessing(clusters, boxes, polar_img, masked_img, theta_fixed, debug=debug)
    else:
        for cluster in clusters:
            if cluster.type == 'rings':
                fitting_result = fit_ring_cluster(cluster, boxes, masked_img, peaks_pool, debug)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)
            elif cluster.type == 'peaks':
                fitting_result = fit_peak_cluster(cluster, boxes, polar_img, peaks_pool, theta_fixed, debug)
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)
        for cluster in clusters:
            if cluster.type == 'both':
                fitting_result = fit_peak_on_ring_cluster(cluster, boxes, polar_img, peaks_pool, theta_fixed, debug)
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



class ProcessDataFromFile:
    """
    Process data from a NeXus file with clustering and Gaussian fitting.

    This class reads a NeXus file (converted images with detected boxes),
    processes a specific entry and frame (or all if None), performs clustering
    of peaks and rings, and fits the clusters with a set of one- or
    two-dimensional Gaussian functions.

    Parameters
    ----------
    filename : str
        Path to the NeXus file containing the data.
    entry : str, optional
        Specific entry in the file to process. If None, all entries are processed.
    frame_num : int, optional
        Frame number to process. If None, all frames are processed.
    crit_angle : float, optional
        Critical angle to shift the sample horizon (in degrees). Default is 0.0.
    polar_shape : Any, optional
        Shape of the polar array used for analysis. Default is [512, 1024].
    clustering_distance_peaks : float, optional
        Maximum distance (in pixels) for clustering peaks. Default is 10.0.
    clustering_distance_rings : float, optional
        Maximum distance (in pixels) for clustering rings. Default is 10.0.
    clustering_extend : float, optional
        Number of pixels to extend the cluster size. Default is 2.0.
    use_pool : bool, optional
        Whether to use peak pool from the previous image. Default is False.
    debug : bool, optional
        Enable debug mode to plot fitting results and parameters. Default is False.
    theta_fixed : bool, optional
        Whether to fix Gaussian tilt angle to 0° (azimuthal direction) during fitting. Default is True
    """

    def __init__(
        self,
        filename: str,
        entry: Optional[str] = None,
        frame_num: Optional[int] = None,
        crit_angle: float = 0.0,
        polar_shape: Any = None,
        clustering_distance_peaks: float = 10.0,
        clustering_distance_rings: float = 10.0,
        clustering_extend: float = 2.0,
        use_pool: bool = False,
        debug: bool = False,
        multiprocessing: bool = False,
        theta_fixed: bool = True
    ):
        self.filename = filename
        self.entry = entry
        self.frame_num = frame_num
        self.crit_angle = crit_angle
        self.polar_shape = polar_shape if polar_shape is not None else [512, 1024]
        self.clustering_distance_peaks = clustering_distance_peaks
        self.clustering_distance_rings = clustering_distance_rings
        self.clustering_extend = clustering_extend
        self.use_pool = use_pool
        self.debug = debug
        self.multiprocessing = multiprocessing
        self.theta_fixed = theta_fixed

        self.process_data_from_file()

    def process_data_from_file(self):
        """
        Process all entries or a specific entry from the NeXus file.

        This method opens the NeXus file, retrieves available entries, and
        processes either the specified entry or all entries sequentially.
        """
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
        """
        Process a single entry from the NeXus file.

        Initializes internal state and processes either all frames or a specific frame.

        Parameters
        ----------
        entry : str
            Name of the entry to process.
        """
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
        """
        Process a single frame of an entry: preprocess the image and detected peaks,
        perform clustering, and fit Gaussian functions.

        Parameters
        ----------
        entry : str
            Name of the NeXus entry to process.
        frame_num : int
            Index of the frame within the entry to process.

        Notes
        -----
        - Loads the converted image and parameters from the NeXus file.
        - Applies preprocessing.
        - Converts the image to polar coordinates.
        - Loads previously detected peaks from the NeXus file.
        - Performs Gaussian fitting and clustering on peaks and rings.
        - Updates peak pools if use_pool is enabled.
        - Saves the fit results back into the NeXus file.
        """
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
                                     wavelength, np.nanmax(q_xy), np.nanmax(q_z), np.sqrt(np.nanmax(q_z)**2+np.nanmax(q_xy)**2),
                                     self.ang_deg_max,
                                     self.clustering_distance_peaks,
                                     self.clustering_distance_rings, self.clustering_extend, self.theta_fixed,
                                     self.debug, self.multiprocessing, self.peaks_pool)

        img_container_fit.score = detected_peaks['score']
        img_container_fit.visibility = detected_peaks['visibility']
        img_container_fit.q_xy = q_xy
        img_container_fit.q_z = q_z

        img_container_fit.metadata = _set_fitting_metadata(
            clustering_distance_peaks=self.clustering_distance_peaks,
            clustering_distance_rings=self.clustering_distance_rings,
            clustering_extend=self.clustering_extend,
            use_pool=self.use_pool)

        # save img_container_fit
        save_fit(self.filename, entry, img_container_fit, frame_num)

def _set_fitting_metadata(**kwargs):
    metadata = {'program': 'pygidfit',
                'version': importlib.metadata.version("pygidfit"),
                'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                }
    metadata.update(kwargs)
    return metadata


def fit_data(polar_img, radius, radius_width, angle, angle_width, wavelength, q_xy_max, q_z_max, q_abs_max, ang_deg_max = 90,
             clustering_distance_peaks = 10,
             clustering_distance_rings = 10, clustering_extend = 2, theta_fixed = False,
             debug = False, multiprocessing = False, peaks_pool = None):
    """
    Fit detected peaks in a polar image with Gaussian functions and cluster them.

    This function takes a polar-converted image and detected peaks (with their
    radii and angles), preprocesses the detected boxes, performs clustering, and applies
    Gaussian fitting to each cluster. Optionally, it updates a pool of peaks for
    sequential frame processing.

    Parameters
    ----------
    polar_img : ndarray
        Polar-converted 2D image to fit.
    radius : array-like
        Radii of the detected peaks.
    radius_width : array-like
        Widths of the detected peaks in radial direction.
    angle : array-like
        Angular positions of the detected peaks (in degrees).
    angle_width : array-like
        Angular widths of the detected peaks (in degrees).
    wavelength : float
        Wavelength used for the diffraction experiment.
    q_xy_max : float
        Maximum in-plane scattering vector.
    q_z_max : float
        Maximum out-of-plane scattering vector.
    q_abs_max : float
        Maximum absolute scattering vector.
    ang_deg_max : float, optional
        Maximum angle for polar image grid (default is 90).
    clustering_distance_peaks : float, optional
        Maximum distance (in pixels) for clustering peaks (default is 10).
    clustering_distance_rings : float, optional
        Maximum distance (in pixels) for clustering rings (default is 10).
    clustering_extend : float, optional
        Number of pixels to extend cluster size (default is 2).
    theta_fixed : bool, optional
        Whether to fix Gaussian tilt angle to 0° (azimuthal direction) during fitting. Default is True
    debug : bool, optional
        Enable debug mode with plots (default is False).
    multiprocessing : bool, optional
        Use multiprocessing for fitting (default is False).
    peaks_pool : list or None, optional
        Pool of peaks from previous frames; allows sequential fitting updates (default is None).

    Returns
    -------
    img_container : object
        Container object with fitted peak results, metadata, and image info.
    updated_peaks_pool : list or None
        Updated peaks pool if `peaks_pool` was provided, otherwise None.
    """
    polar_shape = polar_img.shape
    # boxes preprocessing
    detected_peaks = DetectedPeaks(radius = radius, radius_width = radius_width, angle = angle, angle_width = angle_width)
    boxes = boxes_preprocessing(detected_peaks,
                        polar_shape, wavelength,
                        q_abs_max,
                        q_xy_max, q_z_max)
    # clustarization
    clusters = cluster_boxes_by_centers(boxes, clustering_distance_peaks, clustering_distance_rings, clustering_extend)

    # real calling of fitting
    fit_single_image(polar_img, boxes, clusters, theta_fixed=theta_fixed, peaks_pool=peaks_pool, debug=debug,
                     multiprocessing=multiprocessing)

    img_container = _data2container(boxes, polar_shape, q_abs_max, ang_deg_max,
                                    q_xy_max, q_z_max,
                                    wavelength)
    if peaks_pool is None:
        return img_container, None
    else:
        return img_container, boxes

