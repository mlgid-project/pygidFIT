import numpy as np
from lmfit import Model, Parameters
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Ellipse, Arc
from numba import njit
from multiprocessing import Pool, freeze_support
import time

from pygidfit.box_utils import (
    make_box_attributes,
)



def gaussian_height(x, radius, amplitude, radius_width):
    return amplitude * np.exp(-0.5 * ((x - radius) / radius_width) ** 2)

def safe_center_of_mass(arr):
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.nan, np.nan
    norm = np.nansum(arr)
    if norm == 0:
        return np.nan, np.nan
    y, x = np.indices(arr.shape)
    com_y = np.nansum(y * arr) / norm
    com_x = np.nansum(x * arr) / norm
    return com_y, com_x

@njit
def two_d_rotated_gaussian(x, y, amp, xo, yo, sigma_x, sigma_y, theta):
    x0 = float(xo)
    y0 = float(yo)

    if theta == 0.0:
        a = 1.0 / (2.0 * sigma_x ** 2)
        b = 0.0
        c = 1.0 / (2.0 * sigma_y ** 2)
    else:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        a = (cos_t ** 2) / (2 * sigma_x ** 2) + (sin_t ** 2) / (2 * sigma_y ** 2)
        b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
        c = (sin_t ** 2) / (2 * sigma_x ** 2) + (cos_t ** 2) / (2 * sigma_y ** 2)

    return amp * np.exp(-(a * (x - x0) ** 2 + 2.0 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

# def two_d_gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta):
#     """2D rotated Gaussian function."""
#     x0 = float(xo)
#     y0 = float(yo)
#     if theta == 0:
#         a = 1 / (2 * sigma_x ** 2)
#         c = 1 / (2 * sigma_y ** 2)
#         b = 0
#     else:
#         a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
#         b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
#         c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
#     result = amplitude * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))
#     return result

# @njit
def sum_of_gaussians_and_plane(x, y, param_array, n):
    z = np.zeros_like(x)

    for i in range(n):
        base = i * 6
        amp = param_array[base]
        xo = param_array[base + 1]
        yo = param_array[base + 2]
        sigx = param_array[base + 3]
        sigy = param_array[base + 4]
        theta = param_array[base + 5]

        if np.isnan([amp, xo, yo, sigx, sigy, theta]).any():
            continue

        z += two_d_rotated_gaussian(x, y, amp, xo, yo, sigx, sigy, theta)

    a = param_array[n * 6] if n * 6 < len(param_array) else 0
    b = param_array[n * 6 + 1] if n * 6 + 1 < len(param_array) else 0
    c = param_array[n * 6 + 2] if n * 6 + 2 < len(param_array) else 0
    z += a * x + b * y + c

    return z


# def sum_of_gaussians_and_plane(x, y, param_array, n):
#     z = np.zeros_like(x)
#     for i in range(n):
#         base = i * 6
#         amp = param_array[base]
#         xo = param_array[base + 1]
#         yo = param_array[base + 2]
#         sigx = param_array[base + 3]
#         sigy = param_array[base + 4]
#         theta = param_array[base + 5]
#         z += two_d_rotated_gaussian(x, y, amp, xo, yo, sigx, sigy, theta)
#
#     a = param_array[n * 6]
#     b = param_array[n * 6 + 1]
#     c = param_array[n * 6 + 2]
#     z += a * x + b * y + c
#     return z
def build_sum_gaussians_wrapper(n):
    def model_func(x, y, **params):
        param_list = []
        for i in range(n):
            for key in ['amplitude', 'radius', 'angle', 'radius_width', 'angle_width', 'theta']:
                param_list.append(params.get(f'g{i}_{key}', np.nan))
        param_list.extend([
            params.get('A', np.nan),
            params.get('B', np.nan),
            params.get('C', np.nan)
        ])
        param_array = np.array(param_list, dtype=np.float64)
        return sum_of_gaussians_and_plane(x, y, param_array, n)
    return model_func

# def build_sum_gaussians_wrapper(n):
#     def model_func(x, y, **params):
#         param_list = []
#         for i in range(n):
#             for key in ['amplitude', 'radius', 'angle', 'radius_width', 'angle_width', 'theta']:
#                 param_list.append(params[f'g{i}_{key}'])
#         param_list.extend([params['A'], params['B'], params['C']])
#         param_array = np.array(param_list, dtype=np.float64)
#         return sum_of_gaussians_and_plane(x, y, param_array, n)
#     return model_func


# def build_sum_gaussians(n):
#     """Constructs a model function that sums N 2D Gaussians + a linear background plane."""
#
#     def model_func(x, y, **params):
#         z = np.zeros_like(x)
#         for i in range(n):
#             p = {
#                 k: params[f'{k}{i}'] for k in ['amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta']
#             }
#             z += two_d_gaussian(x, y, **p, offset=0)
#         # Add linear background plane
#         z += params['a'] * x + params['b'] * y + params['c']
#         return z
#
#     return model_func

def fit_peak_cluster(cluster, boxes, img, debug=False):
    """Fit a cluster of 2D Gaussian peaks with a background plane over the bounding box."""
    # Extract ROI bounding box from the cluster
    time0 = time.time()

    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    h, w = img.shape
    xmin = np.clip(xmin, 0, w)
    xmax = np.clip(xmax, 0, w)
    ymin = np.clip(ymin, 0, h)
    ymax = np.clip(ymax, 0, h)

    # Extract ROI from the image
    roi = img[ymin:ymax, xmin:xmax]
    y_len, x_len = roi.shape

    # Create grid coordinates
    Y, X = np.mgrid[0:y_len, 0:x_len]
    # Y, X = np.mgrid[ymin:ymax, xmin:xmax]
    data = roi.ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    X_flat = X_flat[valid_mask]
    Y_flat = Y_flat[valid_mask]

    # Plane params
    a, b, c = 0, 0, np.nanpercentile(roi, 10)
    X_plane, Y_plane = np.mgrid[0:y_len, 0:x_len]
    # X_plane, Y_plane = np.mgrid[ymin: ymax, xmin: xmax]
    Z_plane = a * X_plane + b * Y_plane + c
    roi_corrected = roi - Z_plane

    # Build sum of 2D Gaussians model
    # model = Model(build_sum_gaussians(len(cluster.indices)), independent_vars=['x', 'y'])
    model = Model(build_sum_gaussians_wrapper(len(cluster.indices)), independent_vars=['x', 'y'])

    params = Parameters()

    # Loop through all boxes in the cluster
    for i, idx in enumerate(cluster.indices):
        box = boxes[idx]
        # Convert box limits to ROI-relative pixel indices
        x0 = int(np.clip(np.round(box.limits[0]) - xmin, 0, x_len - 1))
        y0 = int(np.clip(np.round(box.limits[1]) - ymin, 0, y_len - 1))
        x1 = int(np.clip(np.round(box.limits[2]) - xmin, 0, x_len - 1))
        y1 = int(np.clip(np.round(box.limits[3]) - ymin, 0, y_len - 1))
        sub = roi_corrected[y0:y1, x0:x1]

        # Skip empty or invalid boxes
        if sub.size == 0 or not np.isfinite(sub).any():
            # continue
            amp = 0
            xo = (x0 + x1) / 2
            yo = (y0 + y1) / 2
            sigma_x = max((x1 - x0) / 2 / 2.355, 1.0)
            sigma_y = max((y1 - y0) / 2 / 2.355, 1.0)
        else:
        # Estimate initial parameters
            amp = np.nanmax(sub)
            yy, xx = np.indices(sub.shape)
            com_y, com_x = safe_center_of_mass(sub)
            xo = x0 + com_x
            yo = y0 + com_y
            sigma_x = max((x1 - x0) /2/2.355, 1.0)  # FWHM to sigma conversion
            sigma_y = max((y1 - y0) /2/2.355, 1.0)

        x_bound_min = np.clip(x0 - (x1 - x0)/2 + xmin, 0, w) - xmin
        x_bound_max = np.clip(x1 + (x1 - x0)/2 + xmax, 0, w) - xmax
        y_bound_min = np.clip(y0 - (y1 - y0)/2 + ymin, 0, h) - ymin
        y_bound_max = np.clip(y1 + (y1 - y0) / 2 + ymin, 0, h) - ymin if not box.is_cut_qz else h - ymin

        # Add Gaussian parameters to the model
        params.add(f'g{i}_amplitude', value=amp, min=0)
        params.add(f'g{i}_radius', value=xo, min=x_bound_min, max=x_bound_max)
        params.add(f'g{i}_angle', value=yo, min=y_bound_min, max=y_bound_max)
        params.add(f'g{i}_radius_width', value=sigma_x, min=(x1-x0)/8, max = (x1-x0)/2)
        params.add(f'g{i}_angle_width', value=sigma_y, min=(y1-y0)/8, max = (y1-y0)/2)
        params.add(f'g{i}_theta', value=0, vary=False)

    # Add parameters for the background plane
    params.add('A', value=a)
    params.add('B', value=b)
    params.add('C', value=c)

    # Exit if no valid peaks
    if len(params) == 0:
        return None, None, None
    time1 = time.time()

    # Perform the fit
    try:
        result = model.fit(data, params=params, x=X_flat, y=Y_flat,) # max_nfev=100
        if debug:
            print("result.success", result.success)
            for name, par in result.params.items():
                print(name, "value:", par.value, "vary:", par.vary, "stderr:", par.stderr)

    except:
        print("Failed")
        print("params", params)
        print("X_flat, Y_flat",X_flat, Y_flat)
    time2 = time.time()

    if debug:
        plot_peak_cluster_debug(
            roi=roi,
            xmin=xmin,
            ymin=ymin,
            cluster=cluster,
            boxes=boxes,
            params=params,
            result=result,
            time_preproc=(time1 - time0),
            time_fit=(time2 - time1)
        )

    list_to_return = {
        'params': dict(result.best_values),
        'errors': {
            name: (result.params[name].stderr if result.params[name].stderr is not None else np.nan)
            for name in result.params
        },
        'success': result.success,
        'message': result.message,
    }

    for i in range(len(cluster.indices)):
        list_to_return['params'][f'g{i}_radius'] +=xmin
        list_to_return['params'][f'g{i}_angle'] += ymin
    list_to_return['params']['C'] -= list_to_return['params']['A']*xmin + list_to_return['params']['B']*ymin
    return list_to_return


def plot_peak_cluster_debug(roi, xmin, ymin, cluster, boxes, params, result, time_preproc, time_fit):
    """
    Visualizes a ROI with bounding boxes and ellipses for initial guesses and fitted Gaussians.

    Parameters
    ----------
    roi : np.ndarray
        Region of Interest (image section).
    xmin, ymin : float
        ROI offset relative to the original image.
    cluster : object
        Cluster object containing `indices`.
    boxes : list
        List of box objects (with `.limits` attribute).
    params : lmfit.Parameters
        Initial Gaussian parameters.
    result : lmfit.ModelResult
        Fit results.
    time_preproc, time_fit : float
        Preprocessing and fitting times in seconds.
    """
    fig, axes = plt.subplots(figsize=(6, 6))
    norm = LogNorm(vmin=np.nanmin(roi[roi > 0]), vmax=np.nanmax(roi))
    axes.imshow(roi, cmap='viridis', origin='lower', norm=norm)

    # Draw boxes
    for i in cluster.indices:
        box = boxes[i]
        x = box.limits[0] - xmin
        y = box.limits[1] - ymin
        w = box.limits[2] - box.limits[0]
        h = box.limits[3] - box.limits[1]
        rect = Rectangle((x, y), w, h, linewidth=5, edgecolor='red',
                         facecolor='None', alpha=1)
        axes.add_patch(rect)
    axes.set_title(str(cluster.indices))

    # Initial peaks
    for i, idx in enumerate(cluster.indices):
        amp = params.get(f'g{i}_amplitude', None)
        xo = params.get(f'g{i}_radius', None)
        yo = params.get(f'g{i}_angle', None)
        sigma_x = params.get(f'g{i}_radius_width', None)
        sigma_y = params.get(f'g{i}_angle_width', None)

        if None in [amp, xo, yo, sigma_x, sigma_y]:
            continue

        ellipse = Ellipse(
            (xo.value, yo.value),
            width=2 * sigma_x.value,
            height=2 * sigma_y.value,
            edgecolor='white',
            facecolor='none',
            linewidth=2,
            alpha=1,
            linestyle='--',
            label=f'init peak {idx}'
        )
        theta = params.get(f'g{i}_theta', None)
        if theta is not None:
            ellipse.angle = np.degrees(theta.value)
        axes.add_patch(ellipse)

    # Fitted peaks
    for i in range(len(cluster.indices)):
        try:
            amp = result.params[f'g{i}_amplitude']
            xo = result.params[f'g{i}_radius']
            yo = result.params[f'g{i}_angle']
            sigma_x = result.params[f'g{i}_radius_width']
            sigma_y = result.params[f'g{i}_angle_width']
            theta = result.params[f'g{i}_theta']
        except KeyError:
            continue

        ellipse_fit = Ellipse(
            (xo.value, yo.value),
            width=2 * sigma_x.value,
            height=2 * sigma_y.value,
            angle=np.degrees(theta.value),
            edgecolor='green',
            facecolor='none',
            linewidth=3,
            alpha=1,
            linestyle='--',
            label=f'fit peak {cluster.indices[i]}'
        )
        axes.add_patch(ellipse_fit)

    axes.set_aspect('auto')
    axes.legend()
    plt.show()

    print(f"Preprocessing took {time_preproc * 1000:.2f} ms")
    print(f"Fitting took {time_fit * 1000:.2f} ms")

def sum_of_gaussians_and_plane_and_1d(x, y, param_array, n, m):
    # z = np.zeros_like(x)
    z = np.zeros_like(x, dtype=np.float64)
    # 2D rotated Gaussians
    for i in range(n):
        base = i * 6
        amp = param_array[base]
        xo = param_array[base + 1]
        yo = param_array[base + 2]
        sigx = param_array[base + 3]
        sigy = param_array[base + 4]
        theta = param_array[base + 5]
        z += two_d_rotated_gaussian(x, y, amp, xo, yo, sigx, sigy, theta)

    # Plane background
    offset_plane = n * 6
    a = param_array[offset_plane]
    b = param_array[offset_plane + 1]
    c = param_array[offset_plane + 2]
    z += a * x + b * y + c

    # 1D Gaussians along x
    offset_1d = offset_plane + 3
    for j in range(m):
        base = offset_1d + j * 3
        amp_1d = param_array[base]
        center_1d = param_array[base + 1]
        sigma_1d = param_array[base + 2]
        z += amp_1d * np.exp(-0.5 * ((x - center_1d) / sigma_1d) ** 2)

    return z

def gaussian_height(x, radius, amplitude, radius_width):
    return amplitude * np.exp(-0.5 * ((x - radius) / radius_width) ** 2)


def build_sum_gaussians_and_1d_wrapper(n, m):
    def model_func(x, y, **params):
        param_list = []
        # 2D Gaussians
        for i in range(n):
            for key in ['amplitude', 'radius', 'angle', 'radius_width', 'angle_width', 'theta']:
                param_list.append(params[f'g{i}_{key}'])

        # Background plane (MUST come before 1D Gaussians)
        param_list.extend([params['A'], params['B'], params['C']])

        # 1D Gaussians (only X-dependent)
        for i in range(m):
            for key in ['amplitude', 'radius', 'radius_width']:
                param_list.append(params[f'g1d_{i}_{key}'])

        param_array = np.array(param_list, dtype=np.float64)
        return sum_of_gaussians_and_plane_and_1d(x, y, param_array, n, m)

    return model_func

def visualize_fit_3d(X, Y, Z_data, Z_fit):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z_data, cmap='viridis', alpha=0.5, rstride=1, cstride=1, edgecolor='none')
    ax.plot_surface(X, Y, Z_fit, cmap='inferno', alpha=0.5, rstride=1, cstride=1, edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Fit vs Data')
    plt.tight_layout()
    plt.show()



def fit_peak_on_ring_cluster(cluster, boxes, img, debug = False):
    time0 = time.time()
    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    h, w = img.shape
    xmin = np.clip(xmin, 0, w)
    xmax = np.clip(xmax, 0, w)
    ymin = np.clip(ymin, 0, h)
    ymax = np.clip(ymax, 0, h)

    # Extract ROI from the image
    roi = img[ymin:ymax, xmin:xmax]
    y_len, x_len = roi.shape

    # Create grid coordinates
    Y, X = np.mgrid[0:y_len, 0:x_len]
    # Y, X = np.mgrid[ymin:ymax, xmin:xmax]
    data = roi.ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    X_flat = X_flat[valid_mask]
    Y_flat = Y_flat[valid_mask]

    # Plane params
    a, b, c = 0, 0, np.nanpercentile(roi, 10)
    X_plane, Y_plane = np.mgrid[0:y_len, 0:x_len]
    # X_plane, Y_plane = np.mgrid[ymin: ymax, xmin: xmax]
    Z_plane = a * X_plane + b * Y_plane + c
    roi_corrected = roi - Z_plane

    # Build sum of 2D Gaussians model



    peak_indices = []
    ring_indices = []
    for i in cluster.indices:
        if boxes[i].is_ring:
            ring_indices.append(i)
        else:
            peak_indices.append(i)

    model = Model(build_sum_gaussians_and_1d_wrapper(len(peak_indices), len(ring_indices)), independent_vars=['x', 'y'])
    params = Parameters()


    # Loop through all boxes in the cluster
    for i, idx in enumerate(peak_indices):
        box = boxes[idx]
        # Convert box limits to ROI-relative pixel indices
        x0 = int(np.clip(np.round(box.limits[0]) - xmin, 0, x_len - 1))
        y0 = int(np.clip(np.round(box.limits[1]) - ymin, 0, y_len - 1))
        x1 = int(np.clip(np.round(box.limits[2]) - xmin, 0, x_len - 1))
        y1 = int(np.clip(np.round(box.limits[3]) - ymin, 0, y_len - 1))
        sub = roi_corrected[y0:y1, x0:x1]


        if sub.size == 0 or not np.isfinite(sub).any():
            # continue
            amp = 0
            xo = (x0 + x1) / 2
            yo = (y0 + y1) / 2
            sigma_x = max((x1 - x0) / 2 / 2.355, 1.0)
            sigma_y = max((y1 - y0) / 2 / 2.355, 1.0)
        else:
        # Estimate initial parameters
            amp = np.nanmax(sub)
            yy, xx = np.indices(sub.shape)
            com_y, com_x = safe_center_of_mass(sub)
            xo = x0 + com_x
            yo = y0 + com_y
            sigma_x = max((x1 - x0) /2/2.355, 1.0)  # FWHM to sigma conversion
            sigma_y = max((y1 - y0) /2/2.355, 1.0)

        x_bound_min = np.clip(x0 - (x1 - x0) / 2 + xmin, 0, w) - xmin
        x_bound_max = np.clip(x1 + (x1 - x0) / 2 + xmin, 0, w) - xmin
        y_bound_min = np.clip(y0 - (y1 - y0) / 2 + ymin, 0, h) - ymin
        y_bound_max = np.clip(y1 + (y1 - y0) / 2 + ymin, 0, h) - ymin if not box.is_cut_qz else h - ymin



        # Add Gaussian parameters to the model
        params.add(f'g{i}_amplitude', value=amp, min=0)
        params.add(f'g{i}_radius', value=xo, min=x_bound_min, max=x_bound_max)
        params.add(f'g{i}_angle', value=yo, min=y_bound_min, max=y_bound_max)
        params.add(f'g{i}_radius_width', value=sigma_x, min=(x1 - x0) / 8, max=(x1 - x0) / 2)
        params.add(f'g{i}_angle_width', value=sigma_y, min=(y1 - y0) / 8, max=(y1 - y0) / 2)
        params.add(f'g{i}_theta', value=0, vary=False)

    # Add parameters for the background plane
    params.add('A', value=a,)
    params.add('B', value=b,)
    params.add('C', value=c)

    for j, idx in enumerate(ring_indices):
        box = boxes[idx]

        center = box.fitting_result['radius']
        sigma = box.fitting_result['radius_width']
        amp = box.fitting_result['amplitude']

        params.add(f'g1d_{j}_amplitude', value=amp, min=0)
        params.add(f'g1d_{j}_radius', value=center-xmin, vary=False)
        params.add(f'g1d_{j}_radius_width', value=sigma, vary=False)


    # Exit if no valid peaks
    if len(params) == 0:
        return None, None, None
    time1 = time.time()

    # Perform the fit
    result = model.fit(data, params=params, x=X_flat, y=Y_flat, )  # max_nfev=100
    time2 = time.time()

    if debug:
        plot_peak_on_ring_cluster_debug(
            X=X,
            Y=Y,
            roi=roi,
            X_flat=X_flat,
            Y_flat=Y_flat,
            xmin=xmin,
            ymin=ymin,
            model=model,
            result=result,
            peak_indices=peak_indices,
            ring_indices=ring_indices,
            cluster=cluster,
            boxes=boxes,
            params=params,
            time_preproc=(time1 - time0),
            time_fit=(time2 - time1),
            visualize_fit_3d_func=visualize_fit_3d
        )

    list_to_return = {
        'params': dict(result.best_values),
        'errors': {
            name: (result.params[name].stderr if result.params[name].stderr is not None else np.nan)
            for name in result.params
        },
        'success': result.success,
        'message': result.message,
    }

    for i in range(len(peak_indices)):
        list_to_return['params'][f'g{i}_radius'] +=xmin
        list_to_return['params'][f'g{i}_angle'] += ymin

    return list_to_return

def plot_peak_on_ring_cluster_debug(X, Y, roi, X_flat, Y_flat, xmin, ymin,
                   model, result, peak_indices, ring_indices,
                   cluster, boxes, params, time_preproc, time_fit,
                   visualize_fit_3d_func):
    """
    Visualizes the ROI with bounding boxes for peaks and rings,
    along with initial guesses and fitted Gaussian ellipses.

    Parameters
    ----------
    X, Y : np.ndarray
        Meshgrid arrays for the ROI.
    roi : np.ndarray
        Region of Interest (image section).
    X_flat, Y_flat : np.ndarray
        Flattened coordinates for model evaluation.
    xmin, ymin : float
        ROI offset relative to the original image.
    model : lmfit.Model
        The fitted model.
    result : lmfit.ModelResult
        Fit results containing parameters.
    peak_indices : list[int]
        Indices of detected peaks.
    ring_indices : list[int]
        Indices of detected rings.
    cluster : object
        Cluster object containing `.indices`.
    boxes : list
        List of box objects (with `.limits` attribute).
    params : lmfit.Parameters
        Initial Gaussian parameters.
    time_preproc, time_fit : float
        Preprocessing and fitting times in seconds.
    visualize_fit_3d_func : callable
        Function for 3D visualization, signature: (X, Y, roi, Z_fit_full).
    """
    # Model evaluation
    Z_fit_valid = model.eval(params=result.params, x=X_flat, y=Y_flat)
    Z_fit_full = np.full_like(roi, np.nan, dtype=np.float64)
    Z_fit_full[np.isfinite(roi)] = Z_fit_valid

    # Optional 3D visualization
    visualize_fit_3d_func(X, Y, roi, Z_fit_full)

    fig, axes = plt.subplots(figsize=(6, 6))
    norm = LogNorm(vmin=np.nanmin(roi[roi > 0]), vmax=np.nanmax(roi))
    axes.imshow(roi, cmap='viridis', origin='lower', norm=norm)

    # Peak bounding boxes
    for i in peak_indices:
        box = boxes[i]
        x = box.limits[0] - xmin
        y = box.limits[1] - ymin
        w = box.limits[2] - box.limits[0]
        h = box.limits[3] - box.limits[1]
        rect = Rectangle((x, y), w, h, linewidth=5, edgecolor='red',
                         facecolor='None', alpha=1)
        axes.add_patch(rect)

    # Ring bounding boxes
    for i in ring_indices:
        box = boxes[i]
        x = box.limits[0] - xmin
        y = box.limits[1] - ymin
        w = box.limits[2] - box.limits[0]
        h = box.limits[3] - box.limits[1]
        rect = Rectangle((x, y), w, h, linewidth=5, edgecolor='black',
                         facecolor='None', alpha=1, linestyle='--',
                         label=f'ring box {i}')
        axes.add_patch(rect)

    axes.set_title(str(cluster.indices))

    # Initial Gaussian ellipses
    for i, idx in enumerate(cluster.indices):
        amp = params.get(f'g{i}_amplitude', None)
        xo = params.get(f'g{i}_radius', None)
        yo = params.get(f'g{i}_angle', None)
        sigma_x = params.get(f'g{i}_radius_width', None)
        sigma_y = params.get(f'g{i}_angle_width', None)

        if None in [amp, xo, yo, sigma_x, sigma_y]:
            continue

        ellipse = Ellipse(
            (xo.value, yo.value),
            width=2 * sigma_x.value,
            height=2 * sigma_y.value,
            edgecolor='white',
            facecolor='none',
            linewidth=2,
            alpha=1,
            linestyle='--',
            label=f'init peak {idx}'
        )
        theta = params.get(f'g{i}_theta', None)
        if theta is not None:
            ellipse.angle = np.degrees(theta.value)

        axes.add_patch(ellipse)

    # Fitted Gaussian ellipses
    for i in range(len(peak_indices)):
        try:
            amp = result.params[f'g{i}_amplitude']
            xo = result.params[f'g{i}_radius']
            yo = result.params[f'g{i}_angle']
            sigma_x = result.params[f'g{i}_radius_width']
            sigma_y = result.params[f'g{i}_angle_width']
            theta = result.params[f'g{i}_theta']
        except KeyError:
            continue

        ellipse_fit = Ellipse(
            (xo.value, yo.value),
            width=2 * sigma_x.value,
            height=2 * sigma_y.value,
            angle=np.degrees(theta.value),
            edgecolor='green',
            facecolor='none',
            linewidth=3,
            alpha=1,
            linestyle='--',
            label=f'fit peak {cluster.indices[i]}'
        )
        axes.add_patch(ellipse_fit)

    axes.set_aspect('auto')
    axes.legend()
    plt.show()

    print(f"Preprocessing took {time_preproc * 1000:.2f} ms")
    print(f"Fitting took {time_fit * 1000:.2f} ms")

def fit_ring_cluster(cluster, boxes, img, debug = False):
    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    h, w = img.shape


    xmin = np.clip(xmin, 0, w)
    xmax = np.clip(xmax, 0, w)
    ymin = np.clip(ymin, 0, h)
    ymax = np.clip(ymax, 0, h)

    roi = img[ymin:ymax, xmin:xmax]
    profile = np.nanmean(roi, axis=0)

    # sum_values = np.nansum(roi, axis=0)
    # count_values = np.sum(~np.isnan(img), axis=0)
    # profile = sum_values / count_values
    # profile[count_values == 0] = np.nan
    # print(sum_values, count_values)
    # plt.plot(profile)
    # plt.show()


    x = np.arange(xmin, xmax)

    # Estimate linear background
    mid = len(profile) // 2
    ind0 = np.nanargmin(profile[:mid])
    ind1 = np.nanargmin(profile[mid:]) + mid
    x0_lin, x1_lin = x[ind0], x[ind1]
    y0_lin, y1_lin = profile[ind0], profile[ind1]
    slope_guess = (y1_lin - y0_lin) / (x1_lin - x0_lin)
    intercept_guess = y0_lin - slope_guess * x0_lin
    background = slope_guess * x + intercept_guess
    profile_corrected = profile - background

    # Background model
    model = LinearModel(prefix='lin_')
    params = model.make_params(intercept=intercept_guess, slope=slope_guess)
    # params['lin_intercept'].set(min=0, max=10)
    # params['lin_slope'].set(min=np.nanmin(profile_corrected), max=5)

    # Add one Gaussian per ring
    for i, idx in enumerate(cluster.indices):
        box = boxes[idx].limits
        x0_box = int(np.round(box[0]))
        x1_box = int(np.round(box[2]))

        x0_box = np.clip(x0_box, 0, w)
        x1_box = np.clip(x1_box, 0, w)

        center_guess = (x0_box + x1_box) / 2
        x0_rel = max(x0_box - xmin, 0)
        x1_rel = min(x1_box - xmin, len(profile_corrected))
        height_guess = np.nanmax(profile_corrected[x0_rel:x1_rel])
        sigma_guess = (x1_box - x0_box) / 4

        gmod = Model(gaussian_height, prefix=f'g{i}_')
        gparams = gmod.make_params(
            radius=center_guess,
            amplitude=height_guess,
            radius_width=sigma_guess
        )
        gparams[f'g{i}_amplitude'].min = 0

        params.update(gparams)
        model += gmod

    try:
        result = model.fit(profile, params, x=x) #, max_nfev=100
        # for i in cluster.indices:
        #     boxes[i].fitted_params = {
        #         k: result.params[k].value for k in result.params
        #     }
        #     boxes[i].fitted_errors = {
        #         k: result.params[k].stderr for k in result.params
        #     }
    except:
        pass

        # for i in cluster.indices:
        #     boxes[i].initial_guess = {
        #         k: params[k].value for k in params
        #     }


    if debug:
        plt.figure(figsize=(6, 4))
        plt.plot(x, profile, 'b', label='Data')
        plt.plot(x, result.best_fit, 'r-', label='Best Fit')
        plt.plot(x, result.init_fit, 'c--', label='Initial Guess')
        plt.title(f"Cluster {cluster.indices.tolist()}")
        plt.xlabel('X [pixels]')
        plt.ylabel('Mean intensity')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if result.success:
        if debug:
            print("succeed")
        param_values = dict(result.best_values)
    else:
        if debug:
            print("failed")
        param_values = {name: result.init_params[name].value for name in result.init_params}

    return {
        'params': param_values,
        'errors': {
            name: (result.params[name].stderr if result.params[name].stderr is not None else np.nan)
            for name in result.params
        },
        'success': result.success,
        'message': result.message,
    }



def process_cluster_args(args):
    cluster, cluster_type, boxes, img, masked_img, debug = args
    if cluster_type == 'rings':
        result = fit_ring_cluster(cluster, boxes, masked_img, debug)
    elif cluster_type == 'peaks':
        result = fit_peak_cluster(cluster, boxes, img, debug)
    elif cluster_type == 'both':
        result = fit_peak_on_ring_cluster(cluster, boxes, img, debug)
    else:
        result = None
    return cluster, result

def fit_clusters_multiprocessing(clusters, boxes, img, masked_img, debug=False):
    cluster_types = ['rings', 'peaks', 'both']

    for ctype in cluster_types:
        ctype_clusters = [(cluster, ctype, boxes, img, masked_img, debug)
                          for cluster in clusters if cluster.type == ctype]
        if not ctype_clusters:
            continue

        with Pool() as pool:
            results = pool.map(process_cluster_args, ctype_clusters)

        for cluster, fitting_result in results:
            make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)
            cluster.fitting_result = fitting_result
