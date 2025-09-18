import numpy as np
from lmfit import Model, Parameters
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Ellipse
from numba import njit
from multiprocessing import Pool, shared_memory
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

def compute_initial_params(sub, x0, y0, x1, y1, debug = False):
    """Compute initial 2D Gaussian parameters from subarray."""
    amp = np.nanmax(sub)
    if debug:
        print("compute_initial_params")
    yy, xx = np.indices(sub.shape)
    com_y, com_x = safe_center_of_mass(sub)
    xo = x0 + com_x
    yo = y0 + com_y
    sigma_x = max((x1 - x0) / 2 / 2.355, 1.0)
    sigma_y = max((y1 - y0) / 2 / 2.355, 1.0)
    return amp, xo, yo, sigma_x, sigma_y


def fit_peak_cluster(cluster, boxes, img, peaks_pool = None, debug=False):
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
    # Y, X = np.mgrid[0:y_len, 0:x_len]
    Y, X = np.mgrid[ymin:ymax, xmin:xmax]
    data = roi.ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    X_flat = X_flat[valid_mask]
    Y_flat = Y_flat[valid_mask]

    # Plane params
    a_bkg, b_bkg, c_bkg = 0, 0, np.nanpercentile(roi, 10)
    # X_plane, Y_plane = np.mgrid[0:y_len, 0:x_len]
    X_plane, Y_plane = np.mgrid[ymin: ymax, xmin: xmax]
    Z_plane = a_bkg * X_plane + b_bkg * Y_plane + c_bkg
    roi_corrected = roi - Z_plane

    # Build sum of 2D Gaussians model
    # model = Model(build_sum_gaussians(len(cluster.indices)), independent_vars=['x', 'y'])
    model = Model(build_sum_gaussians_wrapper(len(cluster.indices)), independent_vars=['x', 'y'])

    params = Parameters()

    # Loop through all boxes in the cluster
    for i, idx in enumerate(cluster.indices):
        box = boxes[idx]
        x0 = int(np.clip(np.round(box.limits[0]), xmin, xmax ))
        y0 = int(np.clip(np.round(box.limits[1]) ,ymin, ymax))
        x1 = int(np.clip(np.round(box.limits[2]) , xmin, xmax))
        y1 = int(np.clip(np.round(box.limits[3]) , ymin, ymax))

        sub = roi_corrected[y0- ymin:y1- ymin, x0- xmin:x1- xmin]

        # Skip empty or invalid boxes
        if sub.size == 0 or not np.isfinite(sub).any():
            if debug:
                print("invalid")
            amp = 0
            xo = (x0 + x1) / 2
            yo = (y0 + y1) / 2
            sigma_x = max((x1 - x0) / 2 / 2.355, 1.0)
            sigma_y = max((y1 - y0) / 2 / 2.355, 1.0)
        else:
            prev_box = None
            if peaks_pool is not None and len(peaks_pool) > 0:
                for b in peaks_pool:
                    r, a = b.fitting_result['radius'], b.fitting_result['angle']
                    # print("box.limits[0], r, box.limits[2],  box.limits[1] , a, box.limits[3] ",box.limits[0], r, box.limits[2],  box.limits[1] , a, box.limits[3])
                    if box.limits[0] <= r <= box.limits[2] and box.limits[1] <= a <= box.limits[3]:
                        prev_box = b
                        break
            if prev_box is not None:
                if debug:
                    print("Use previous peak")
                amp = prev_box.fitting_result['amplitude']
                xo = prev_box.fitting_result['radius']
                yo = prev_box.fitting_result['angle']
                sigma_x = prev_box.fitting_result['radius_width']
                sigma_y = prev_box.fitting_result['angle_width']
            else:
                if debug and peaks_pool is not None:
                    print("Couldn't find previous peak")
                amp, xo, yo, sigma_x, sigma_y = compute_initial_params(sub, x0, y0, x1, y1, debug)

        if False: #debug
            print("amp, xo, yo, sigma_x, sigma_y ", amp, xo, yo, sigma_x, sigma_y )
        x_bound_min = np.clip(x0 - (x1 - x0)/2, xmin, xmax)
        x_bound_max = np.clip(x1 + (x1 - x0)/2 , xmin, xmax)
        y_bound_min = np.clip(y0 - (y1 - y0)/2 , ymin, ymax)
        y_bound_max = np.clip(y1 + (y1 - y0) / 2 , ymin, ymax) if not box.is_cut_qz else h

        # Add Gaussian parameters to the model
        params.add(f'g{i}_amplitude', value=amp, min=0)
        params.add(f'g{i}_radius', value=xo, min=x_bound_min, max=x_bound_max)
        params.add(f'g{i}_angle', value=yo, min=y_bound_min, max=y_bound_max)
        params.add(f'g{i}_radius_width', value=sigma_x, min=(x1-x0)/8, max = (x1-x0)/2)
        params.add(f'g{i}_angle_width', value=sigma_y, min=(y1-y0)/8, max = (y1-y0)/2)
        params.add(f'g{i}_theta', value=0, vary=False)

    # Add parameters for the background plane
    params.add('A', value=a_bkg, min = -0.1, max = 0.1)
    params.add('B', value=b_bkg, min = -1, max = 1)
    params.add('C', value=c_bkg, min = -abs(c_bkg/4)-1, max = abs(c_bkg*2)+1)

    # Exit if no valid peaks
    if len(params) == 0:
        return None, None, None
    time1 = time.time()

    if False:
        res = fit_peak_cluster_jaxfit_from_lmfit(data, X_flat, Y_flat, params)
    if False:
        res = fit_peak_cluster_numba_cuda_from_lmfit(data, X_flat, Y_flat, params, max_nfev=500)

    # Perform the fit
    try:
        result = model.fit(data, params=params, x=X_flat, y=Y_flat, max_nfev=500, method="least_squares") # max_nfev=100
        list_to_return = {
            'params': dict(result.best_values),
            'errors': {
                name: (result.params[name].stderr if result.params[name].stderr is not None else np.nan)
                for name in result.params
            },
            'success': result.success,
            'message': result.message,
        }

    except:
        # print("Failed")
        param_values = {name: p.value for name, p in params.items()}
        param_errors = {name: np.nan for name in params}
        result = None
        list_to_return = {
            'params': param_values,
            'errors': param_errors,
            'success': False,
            'message': 'fit failed'
        }
    time2 = time.time()

    if debug:
        try:
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
        except:
            print("Can't plot peak cluster")
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
    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    axes.imshow(roi, cmap='inferno', origin='lower', norm=norm, extent=[xmin, xmax, ymin, ymax])

    # Draw boxes
    for i in cluster.indices:
        box = boxes[i]
        x = box.limits[0] #- xmin
        y = box.limits[1] #- ymin
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

    ax.plot_surface(X, Y, Z_data, cmap='inferno', alpha=0.5, rstride=1, cstride=1, edgecolor='none')
    ax.plot_surface(X, Y, Z_fit, cmap='inferno', alpha=0.5, rstride=1, cstride=1, edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Fit vs Data')
    plt.tight_layout()
    plt.show()



def fit_peak_on_ring_cluster(cluster, boxes, img, peaks_pool, debug = False):
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
    # Y, X = np.mgrid[0:y_len, 0:x_len]
    Y, X = np.mgrid[ymin:ymax, xmin:xmax]
    data = roi.ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    X_flat = X_flat[valid_mask]
    Y_flat = Y_flat[valid_mask]

    # Plane params
    a, b, c = 0, 0, np.nanpercentile(roi, 10)
    X_plane, Y_plane = np.mgrid[ymin: ymax, xmin: xmax]
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
        x0 = int(np.clip(np.round(box.limits[0]), xmin, xmax ))
        y0 = int(np.clip(np.round(box.limits[1]) ,ymin, ymax))
        x1 = int(np.clip(np.round(box.limits[2]) , xmin, xmax))
        y1 = int(np.clip(np.round(box.limits[3]) , ymin, ymax))

        # sub = roi_corrected[y0:y1, x0:x1]
        sub = roi_corrected[y0 - ymin:y1 - ymin, x0 - xmin:x1 - xmin]

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

        x_bound_min = np.clip(x0 - (x1 - x0)/2, xmin, xmax)
        x_bound_max = np.clip(x1 + (x1 - x0)/2 , xmin, xmax)
        y_bound_min = np.clip(y0 - (y1 - y0)/2 , ymin, ymax)
        y_bound_max = np.clip(y1 + (y1 - y0)/2 , ymin, ymax) if not box.is_cut_qz else h

        # Add Gaussian parameters to the model
        params.add(f'g{i}_amplitude', value=amp, min=0)
        params.add(f'g{i}_radius', value=xo, min=x_bound_min, max=x_bound_max)
        params.add(f'g{i}_angle', value=yo, min=y_bound_min, max=y_bound_max)
        params.add(f'g{i}_radius_width', value=sigma_x, min=(x1 - x0) / 8, max=(x1 - x0) / 2)
        params.add(f'g{i}_angle_width', value=sigma_y, min=(y1 - y0) / 8, max=(y1 - y0) / 2)
        params.add(f'g{i}_theta', value=0, vary=False)

    # params.add('A', value=a, min = -1, max = 1)
    params.add('A', value=a, min=-0.1, max=0.1)
    params.add('B', value=b, min = -1, max = 1)
    params.add('C', value=c, min = -abs(c/4)-1, max = abs(c*2)+1)

    for j, idx in enumerate(ring_indices):
        box = boxes[idx]

        center = box.fitting_result['radius']
        sigma = box.fitting_result['radius_width']
        amp = box.fitting_result['amplitude']

        params.add(f'g1d_{j}_amplitude', value=amp, min=0)
        params.add(f'g1d_{j}_radius', value=center, vary=False)
        params.add(f'g1d_{j}_radius_width', value=sigma, vary=False)


    # Exit if no valid peaks
    if len(params) == 0:
        return None, None, None
    time1 = time.time()

    for name, p in params.items():
        if np.isnan(p.value):
            if "amplitude" in name:
                p.value = 1.0
            else:
                p.value = 0.0

    # Perform the fit
    result = model.fit(data, params=params, x=X_flat, y=Y_flat, max_nfev=500, method="least_squares")  # max_nfev=100
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
    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    axes.imshow(roi, cmap='inferno', origin='lower', norm=norm, extent=[xmin, xmax, ymin, ymax])

    # Peak bounding boxes
    for i in peak_indices:
        box = boxes[i]
        x = box.limits[0] # - xmin
        y = box.limits[1] #- ymin
        w = box.limits[2] - box.limits[0]
        h = box.limits[3] - box.limits[1]
        rect = Rectangle((x, y), w, h, linewidth=5, edgecolor='red',
                         facecolor='None', alpha=1)
        axes.add_patch(rect)

    # Ring bounding boxes
    for i in ring_indices:
        box = boxes[i]
        x = box.limits[0] # box.limits[0] - xmin
        y = ymin # box.limits[1] - ymin
        w = box.limits[2] - box.limits[0]
        h = ymax - ymin# box.limits[3] - box.limits[1]
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

def fit_ring_cluster(cluster, boxes, img,  peaks_pool, debug = False):
    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    h, w = img.shape


    xmin = np.clip(xmin, 0, w)
    xmax = np.clip(xmax, 0, w)
    ymin = np.clip(ymin, 0, h)
    ymax = np.clip(ymax, 0, h)

    roi = img[ymin:ymax, xmin:xmax]
    profile = np.nanmean(roi, axis=0)

    x = np.arange(xmin, xmax)

    # Estimate linear background
    mid = len(profile) // 2

    left = profile[:mid]
    right = profile[mid:]

    if np.all(np.isnan(left)) or np.all(np.isnan(right)):
        ind0 = 0
        ind1 = len(profile) - 1
    else:
        ind0 = np.nanargmin(left)
        ind1 = np.nanargmin(right) + mid

    x0_lin, x1_lin = x[ind0], x[ind1]
    y0_lin, y1_lin = profile[ind0], profile[ind1]
    slope_guess = (y1_lin - y0_lin) / (x1_lin - x0_lin)
    intercept_guess = y0_lin - slope_guess * x0_lin
    if np.isnan(slope_guess) or np.isnan(intercept_guess):
        slope_guess = 0.0
        intercept_guess = 0.0
    background = slope_guess * x + intercept_guess
    profile_corrected = profile - background

    # Background model
    model = LinearModel(prefix='lin_')
    params = model.make_params(intercept=intercept_guess, slope=slope_guess)

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
        if x1_rel <= x0_rel:
            height_guess = 0
            sigma_guess = 1
        else:
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
        mask = np.isfinite(profile)
        profile = profile[mask]
        x =  x[mask]
        result = model.fit(profile, params, x=x, max_nfev=500, method="least_squares",) #, max_nfev=100
        param_values = dict(result.best_values)
        param_errors = {
            name: (result.params[name].stderr if result.params[name].stderr is not None else np.nan)
            for name in result.params
        }
        success = result.success
        message = result.message
    except:
        param_values = {name: p.value for name, p in params.items()}
        param_errors = {name: np.nan for name in params}
        success = False
        message = "Failed"

    if debug:
        plt.figure(figsize=(6, 4))
        plt.plot(x, profile, 'b', label='Data')
        try:
            plt.plot(x, result.best_fit, 'r-', label='Best Fit')
            plt.plot(x, result.init_fit, 'c--', label='Initial Guess')
        except:
            print("Fitting failed")
        plt.title(f"Cluster {cluster.indices.tolist()}")
        plt.xlabel('X [pixels]')
        plt.ylabel('Mean intensity')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'params': param_values,
        'errors': param_errors,
        'success': success,
        'message': message,
    }



def process_cluster_args(args):
    cluster, cluster_type, boxes, img, masked_img, debug = args
    if cluster_type == 'rings':
        result = fit_ring_cluster(cluster, boxes, masked_img, None, debug)
    elif cluster_type == 'peaks':
        result = fit_peak_cluster(cluster, boxes, img, None, debug)
    elif cluster_type == 'both':
        result = fit_peak_on_ring_cluster(cluster, boxes, img, None, debug)
    else:
        result = None
    return cluster, result

##### MP with shared memory
def fit_clusters_multiprocessing(clusters, boxes, img, masked_img, debug=False):
    cluster_types = ['rings', 'peaks', 'both']

    shm_img, shm_masked, img_shape, masked_shape, img_dtype, masked_dtype = init_shared_images(img, masked_img)

    try:
        for ctype in cluster_types:
            ctype_clusters = [(cluster, ctype, boxes,
                               shm_img.name, shm_masked.name,
                               img_shape, masked_shape, img_dtype, masked_dtype,
                               debug)
                              for cluster in clusters if cluster.type == ctype]
            if not ctype_clusters:
                continue

            with Pool() as pool:
                results = pool.map(process_cluster_shared, ctype_clusters)

            for cluster, fitting_result in results:
                make_box_attributes(cluster.indices, boxes, fitting_result, cluster.type, debug)
                cluster.fitting_result = fitting_result
    finally:
        shm_img.close()
        shm_img.unlink()
        shm_masked.close()
        shm_masked.unlink()


def init_shared_images(img, masked_img):
    shm_img = shared_memory.SharedMemory(create=True, size=img.nbytes)
    shm_masked = shared_memory.SharedMemory(create=True, size=masked_img.nbytes)

    shm_img_array = np.ndarray(img.shape, dtype=img.dtype, buffer=shm_img.buf)
    shm_masked_array = np.ndarray(masked_img.shape, dtype=masked_img.dtype, buffer=shm_masked.buf)
    np.copyto(shm_img_array, img)
    np.copyto(shm_masked_array, masked_img)

    return shm_img, shm_masked, img.shape, masked_img.shape, img.dtype, masked_img.dtype

def process_cluster_shared(args):
    cluster, ctype, boxes, shm_img_name, shm_masked_name, img_shape, masked_shape, img_dtype, masked_dtype, debug = args

    existing_shm_img = shared_memory.SharedMemory(name=shm_img_name)
    existing_shm_masked = shared_memory.SharedMemory(name=shm_masked_name)
    img = np.ndarray(img_shape, dtype=img_dtype, buffer=existing_shm_img.buf)
    masked_img = np.ndarray(masked_shape, dtype=masked_dtype, buffer=existing_shm_masked.buf)

    fitting_result = process_cluster_args((cluster, ctype, boxes, img, masked_img, debug))

    return fitting_result


############## numba start

# import numpy as np
# from numba import cuda
# from scipy.optimize import least_squares
# import cupy as cp
# import math
#
# @cuda.jit
# def compute_residuals_kernel(X, Y, zdata, params, residuals, N_gauss):
#     idx = cuda.grid(1)
#     if idx >= zdata.size:
#         return
#
#     val = 0.0
#     for g in range(N_gauss):
#         base = g*6
#         n0 = params[base]
#         x0 = params[base+1]
#         y0 = params[base+2]
#         sigma_x = params[base+3]
#         sigma_y = params[base+4]
#         theta = params[base+5]
#
#         Xr = X[idx] - x0
#         Yr = Y[idx] - y0
#         gauss = math.exp(-0.5*(Xr**2/(sigma_x*sigma_x) + Yr**2/(sigma_y*sigma_y))) * n0
#         val += gauss
#
#     A = params[-3]
#     B = params[-2]
#     C = params[-1]
#     val += A + B*X[idx] + C*Y[idx]
#
#     residuals[idx] = zdata[idx]
#     residuals[idx] -= val
#
# def fit_peak_cluster_numba_cuda_from_lmfit(zdata, X, Y, lmfit_params, max_nfev=500):
#     param_list = []
#     bounds_min = []
#     bounds_max = []
#
#     n_peaks = max([int(name.split('_')[0][1:]) for name in lmfit_params.keys() if name.startswith('g')]) + 1
#
#     for i in range(n_peaks):
#         for key in ['amplitude', 'radius', 'angle', 'radius_width', 'angle_width', 'theta']:
#             pname = f'g{i}_{key}'
#             p = lmfit_params[pname]
#             param_list.append(p.value)
#             bounds_min.append(-np.inf if p.min is None else p.min)
#             bounds_max.append(np.inf if p.max is None else p.max)
#
#     for key in ['A', 'B', 'C']:
#         p = lmfit_params[key]
#         param_list.append(p.value)
#         bounds_min.append(-np.inf if p.min is None else p.min)
#         bounds_max.append(np.inf if p.max is None else p.max)
#
#     p0 = np.array(param_list)
#     bounds = (np.array(bounds_min), np.array(bounds_max))
#     X_gpu = cuda.to_device(X)
#     Y_gpu = cuda.to_device(Y)
#     zdata_gpu = cuda.to_device(zdata)
#     residuals_gpu = cuda.device_array_like(zdata_gpu)
#
#     def compute_residuals(params):
#         params_gpu = cuda.to_device(params)
#         threadsperblock = 128
#         blockspergrid = (zdata_gpu.size + threadsperblock - 1) // threadsperblock
#         compute_residuals_kernel[blockspergrid, threadsperblock](
#             X_gpu, Y_gpu, zdata_gpu, params_gpu, residuals_gpu, n_peaks
#         )
#         return residuals_gpu.copy_to_host()
#
#     numba_time0 = time.time()
#     result = least_squares(compute_residuals, p0, bounds=bounds, max_nfev=max_nfev)
#     # try:
#     #     result = least_squares(compute_residuals, p0, bounds=bounds, max_nfev=max_nfev)
#     # except:
#     #     result = None
#     numba_time1 = time.time()
#     print("!!!!!!!!!!!! numba_time", (numba_time1 - numba_time0) * 1000)
#     # print(f"numba result{result}")
#
#     return result

############## numba start

############## jaxfit start
# import jax
# import jax.numpy as jnp
# from jaxfit import CurveFit
#
# def two_d_rotated_gaussian_jax(x, y, amp, xo, yo, sigx, sigy, theta):
#     cos_t = jnp.cos(theta)
#     sin_t = jnp.sin(theta)
#     a = (cos_t**2)/(2*sigx**2) + (sin_t**2)/(2*sigy**2)
#     b = -jnp.sin(2*theta)/(4*sigx**2) + jnp.sin(2*theta)/(4*sigy**2)
#     c = (sin_t**2)/(2*sigx**2) + (cos_t**2)/(2*sigy**2)
#     return amp * jnp.exp(-(a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))
#
# def sum_of_gaussians_and_plane_jax(x, y, param_array, n):
#     def add_gaussian(i, z):
#         base = i*6
#         amp = param_array[base]
#         xo = param_array[base+1]
#         yo = param_array[base+2]
#         sigx = param_array[base+3]
#         sigy = param_array[base+4]
#         theta = param_array[base+5]
#
#         z += jax.lax.cond(
#             jnp.any(jnp.isnan(jnp.array([amp, xo, yo, sigx, sigy, theta]))),
#             lambda _: jnp.zeros_like(x),
#             lambda _: two_d_rotated_gaussian_jax(x, y, amp, xo, yo, sigx, sigy, theta),
#             operand=None
#         )
#         return z
#
#     z_init = jnp.zeros_like(x)
#     z = jax.lax.fori_loop(0, n, add_gaussian, z_init)
#
#     a_plane = param_array[n*6] if n*6 < len(param_array) else 0
#     b_plane = param_array[n*6+1] if n*6+1 < len(param_array) else 0
#     c_plane = param_array[n*6+2] if n*6+2 < len(param_array) else 0
#     z += a_plane * x + b_plane * y + c_plane
#     return z
#
# def fit_peak_cluster_jaxfit_from_lmfit(data, X_flat, Y_flat, lm_params):
#     param_names = list(lm_params.keys())
#     n_peaks = sum(1 for name in param_names if name.startswith('g') and '_amplitude' in name)
#
#     print("n_peaks", n_peaks)
#
#     param_list = []
#     bounds_min = []
#     bounds_max = []
#
#     for i in range(n_peaks):
#         for key in ['amplitude', 'radius', 'angle', 'radius_width', 'angle_width', 'theta']:
#             pname = f'g{i}_{key}'
#             p = lm_params[pname]
#             param_list.append(p.value)
#             bounds_min.append(-jnp.inf if p.min is None else p.min)
#             bounds_max.append(jnp.inf if p.max is None else p.max)
#
#     for pname in ['A', 'B', 'C']:
#         p = lm_params[pname]
#         param_list.append(p.value)
#         bounds_min.append(-jnp.inf if p.min is None else p.min)
#         bounds_max.append(jnp.inf if p.max is None else p.max)
#
#     params_jax = jnp.array(param_list, dtype=jnp.float32)
#     print("params_jax", params_jax)
#     bounds = (jnp.array(bounds_min, dtype=jnp.float32), jnp.array(bounds_max, dtype=jnp.float32))
#
#     coords_tuple = [jnp.array(X_flat), jnp.array(Y_flat)]
#     jcf = CurveFit()
#
#     def model_for_fit(coords, *param_array):
#         param_array = jnp.array(param_array)
#         return sum_of_gaussians_and_plane_jax(coords[0], coords[1], param_array, n_peaks).ravel()
#
#     jaxfit_time0 = time.time()
#     try:
#         popt, pcov = jcf.curve_fit(model_for_fit, coords_tuple, data, p0=params_jax, bounds=bounds)
#     except:
#         popt, pcov = None, None
#     jaxfit_time1 = time.time()
#     print("jaxfit_time", (jaxfit_time1 - jaxfit_time0) * 1000)
#     print("popt", popt)
#     return popt, pcov

############## jaxfit end

