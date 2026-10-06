import os
import copy
import numpy as np
from lmfit import Model, Parameters
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Ellipse
from numba import jit
from multiprocessing import Pool, shared_memory
import time
from math import sin, cos, exp
import numexpr as ne
from scipy.optimize import least_squares

from pygidfit.box_utils import (
    make_box_attributes,
)

# Analytic Jacobian on/off switch and size cap: past ~6 Gaussian components,
# scipy's own linear algebra for the (n_pixels x n_params) Jacobian matrix
# costs more than the analytic shortcut saves (measured).
_USE_ANALYTIC_JAC = os.environ.get("PYGIDFIT_ANALYTIC_JAC", "1") != "0"
_MAX_ANALYTIC_JAC_COMPONENTS = 6

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


@jit(nopython=True, fastmath=True)
def _gaussian2d_jac_array(x, y, param_array, n):
    """Analytic d(model)/d(param) for n 2D rotated Gaussians + a background
    plane, as a dense (len(x), 6n+3) array, laid out like `sum_of_gaussians_and_plane`'s
    own param_array (amp,xo,yo,sigx,sigy,theta per gaussian, then A,B,C)."""
    npix = x.size
    jac = np.zeros((npix, n * 6 + 3))
    for i in range(n):
        base = i * 6
        amp = param_array[base]
        xo = param_array[base + 1]
        yo = param_array[base + 2]
        sigx = param_array[base + 3]
        sigy = param_array[base + 4]
        theta = param_array[base + 5]

        cos_t = cos(theta)
        sin_t = sin(theta)
        sin2t = 2.0 * sin_t * cos_t
        cos2t = cos_t * cos_t - sin_t * sin_t
        inv_sigx2 = 1.0 / (sigx * sigx)
        inv_sigy2 = 1.0 / (sigy * sigy)

        a = 0.5 * (cos_t * cos_t * inv_sigx2 + sin_t * sin_t * inv_sigy2)
        b = 0.25 * sin2t * (inv_sigy2 - inv_sigx2)
        c = 0.5 * (sin_t * sin_t * inv_sigx2 + cos_t * cos_t * inv_sigy2)

        da_dsigx = -cos_t * cos_t / sigx ** 3
        db_dsigx = 0.5 * sin2t / sigx ** 3
        dc_dsigx = -sin_t * sin_t / sigx ** 3

        da_dsigy = -sin_t * sin_t / sigy ** 3
        db_dsigy = -0.5 * sin2t / sigy ** 3
        dc_dsigy = -cos_t * cos_t / sigy ** 3

        da_dtheta = 0.5 * sin2t * (inv_sigy2 - inv_sigx2)
        db_dtheta = 0.5 * cos2t * (inv_sigy2 - inv_sigx2)
        dc_dtheta = -da_dtheta

        for k in range(npix):
            dx = x.flat[k] - xo
            dy = y.flat[k] - yo
            expo = exp(-(a * dx * dx + 2 * b * dx * dy + c * dy * dy))
            g = amp * expo
            jac[k, base] = expo
            jac[k, base + 1] = g * (2 * a * dx + 2 * b * dy)
            jac[k, base + 2] = g * (2 * b * dx + 2 * c * dy)
            jac[k, base + 3] = g * -(da_dsigx * dx * dx + 2 * db_dsigx * dx * dy + dc_dsigx * dy * dy)
            jac[k, base + 4] = g * -(da_dsigy * dx * dx + 2 * db_dsigy * dx * dy + dc_dsigy * dy * dy)
            jac[k, base + 5] = g * -(da_dtheta * dx * dx + 2 * db_dtheta * dx * dy + dc_dtheta * dy * dy)

    idx_plane = n * 6
    for k in range(npix):
        jac[k, idx_plane] = x.flat[k]
        jac[k, idx_plane + 1] = y.flat[k]
        jac[k, idx_plane + 2] = 1.0
    return jac


def _gaussian2d_terms(params, x, y, n):
    """d(model)/d(param) dict for n 2D rotated Gaussians + a background plane."""
    param_array = np.empty(n * 6 + 3, dtype=np.float64)
    pos = 0
    for i in range(n):
        param_array[pos] = params[f'g{i}_amplitude'].value
        param_array[pos + 1] = params[f'g{i}_radius'].value
        param_array[pos + 2] = params[f'g{i}_angle'].value
        param_array[pos + 3] = params[f'g{i}_radius_width'].value
        param_array[pos + 4] = params[f'g{i}_angle_width'].value
        param_array[pos + 5] = params[f'g{i}_theta'].value
        pos += 6
    param_array[pos] = params['A'].value
    param_array[pos + 1] = params['B'].value
    param_array[pos + 2] = params['C'].value

    full_jac = _gaussian2d_jac_array(x, y, param_array, n)

    derivs = {}
    pos = 0
    for i in range(n):
        derivs[f'g{i}_amplitude'] = full_jac[:, pos]
        derivs[f'g{i}_radius'] = full_jac[:, pos + 1]
        derivs[f'g{i}_angle'] = full_jac[:, pos + 2]
        derivs[f'g{i}_radius_width'] = full_jac[:, pos + 3]
        derivs[f'g{i}_angle_width'] = full_jac[:, pos + 4]
        derivs[f'g{i}_theta'] = full_jac[:, pos + 5]
        pos += 6
    derivs['A'] = full_jac[:, pos]
    derivs['B'] = full_jac[:, pos + 1]
    derivs['C'] = full_jac[:, pos + 2]
    return derivs


def _gaussian1d_terms(params, x, count, prefix):
    """d(model)/d(param) dict for `count` 1D Gaussians along x (ring profiles)."""
    derivs = {}
    for j in range(count):
        amp = params[f'{prefix}{j}_amplitude'].value
        center = params[f'{prefix}{j}_radius'].value
        sigma = params[f'{prefix}{j}_radius_width'].value
        dx = x - center
        expo = np.exp(-0.5 * (dx / sigma) ** 2)
        h = amp * expo
        derivs[f'{prefix}{j}_amplitude'] = expo
        derivs[f'{prefix}{j}_radius'] = h * dx / sigma ** 2
        derivs[f'{prefix}{j}_radius_width'] = h * dx * dx / sigma ** 3
    return derivs


def _params_to_jacobian(params, derivs, weights):
    """Jacobian of the residual (data - model) from a dict of model derivatives."""
    columns = [-derivs[name] for name, par in params.items() if par.vary]
    jac = np.column_stack(columns)
    if weights is not None:
        jac = jac * weights[:, None]
    return jac


def _fit_with_scipy(residual_func, jac_func, params, var_names, use_jac, prefer_lsmr=False):
    """Run scipy.optimize.least_squares directly (bypassing lmfit's own
    per-iteration Parameters/Minimizer bookkeeping). `params` supplies the
    initial values/bounds and is untouched; `residual_func`/`jac_func` take
    the free-parameter vector and update their own closed-over working copy.
    Returns the same {'params','errors','success','message'} shape lmfit's
    Model.fit() used to, with uncertainties computed the same way lmfit does
    (covariance = inv(J^T J) * reduced chi-square).

    `prefer_lsmr=True` tries the iterative 'lsmr' trust-region solver first
    instead of the default 'exact' (SVD-based), falling back to the other on
    failure"""
    x0 = np.array([params[name].value for name in var_names], dtype=float)
    lb = np.array([params[name].min if params[name].min is not None else -np.inf for name in var_names])
    ub = np.array([params[name].max if params[name].max is not None else np.inf for name in var_names])

    def _solve(tr_solver):
        kwargs = dict(
            jac=(jac_func if use_jac else '2-point'),
            bounds=(lb, ub), method="trf",
            ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear', f_scale=1.0,
            max_nfev=500,
        )
        if tr_solver is not None:
            kwargs['tr_solver'] = tr_solver
        return least_squares(residual_func, x0, **kwargs)

    try:
        first_solver, fallback_solver = ('lsmr', None) if prefer_lsmr else (None, 'lsmr')
        result = _solve(first_solver)
        if not result.success:
            # The default 'exact' trust-region subproblem solver occasionally
            # fails to converge where 'lsmr' (iterative) handles it fine, and
            # vice versa -- only retry on an actual failure, so this never
            # touches the (overwhelming majority) success path.
            retry = _solve(fallback_solver)
            if retry.success:
                result = retry

        final_values = dict(zip(var_names, result.x))
        all_values = {name: final_values.get(name, params[name].value) for name in params}

        resid = result.fun
        nfree = len(resid) - len(var_names)
        redchi = float(np.sum(resid ** 2)) / max(1, nfree)

        try:
            jtj = result.jac.T @ result.jac
            # A nearly (but not exactly) singular J^T J doesn't raise on inv() --
            # it silently returns a huge, meaningless "uncertainty". Treat that
            # the same as outright singular.
            if np.linalg.cond(jtj) > 1e10:
                raise np.linalg.LinAlgError("ill-conditioned J^T J")
            cov = np.linalg.inv(jtj) * redchi
            errors = {name: 0.0 for name in params}
            for i, name in enumerate(var_names):
                errors[name] = float(np.sqrt(cov[i, i]))
        except np.linalg.LinAlgError:
            errors = {name: np.nan for name in params}

        return {
            'params': all_values,
            'errors': errors,
            'success': bool(result.success),
            'message': result.message,
        }
    except Exception:
        return {
            'params': {name: p.value for name, p in params.items()},
            'errors': {name: np.nan for name in params},
            'success': False,
            'message': 'fit failed',
        }


class _DebugResultShim:
    """Minimal stand-in for the lmfit ModelResult the debug plotting functions
    expect (they only read result.params[name].value), now that fitting no
    longer goes through lmfit.Model.fit()."""
    def __init__(self, params):
        self.params = params

# @jit(nopython=True, fastmath=True)
def two_d_rotated_gaussian(x, y, amp, xo, yo, sigma_x, sigma_y, theta):
    x0 = xo
    y0 = yo

    cos_t = cos(theta)
    sin_t = sin(theta)

    sin2 = 2.0 * sin_t * cos_t

    a = (cos_t * cos_t) / (2.0 * sigma_x * sigma_x) + (sin_t * sin_t) / (2.0 * sigma_y * sigma_y)
    b = -sin2 / (4.0 * sigma_x * sigma_x) + sin2 / (4.0 * sigma_y * sigma_y)
    c = (sin_t * sin_t) / (2.0 * sigma_x * sigma_x) + (cos_t * cos_t) / (2.0 * sigma_y * sigma_y)
    dx = x - x0
    dy = y - y0
    # out = np.empty_like(x)
    # for i in range(x.size):
    #     val = a * dx.flat[i] * dx.flat[i] + 2.0 * b * dx.flat[i] * dy.flat[i] + c * dy.flat[i] * dy.flat[i]
    #     out.flat[i] = amp * exp(-val)
    out = ne.evaluate("amp * exp(-(a*dx*dx + 2*b*dx*dy + c*dy*dy))")
    return out

@jit(nopython=True, fastmath=True)
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

        cos_t = cos(theta)
        sin_t = sin(theta)
        sin2 = 2.0 * sin_t * cos_t

        a = (cos_t * cos_t) / (2.0 * sigx * sigx) + (sin_t * sin_t) / (2.0 * sigy * sigy)
        b = -sin2 / (4.0 * sigx * sigx) + sin2 / (4.0 * sigy * sigy)
        c = (sin_t * sin_t) / (2.0 * sigx * sigx) + (cos_t * cos_t) / (2.0 * sigy * sigy)

        for k in range(x.size):
            dx = x.flat[k] - xo
            dy = y.flat[k] - yo
            z.flat[k] += amp * exp(-(a*dx*dx + 2*b*dx*dy + c*dy*dy))

    idx_plane = n * 6
    a = param_array[idx_plane] if idx_plane < param_array.size else 0.0
    b = param_array[idx_plane + 1] if idx_plane + 1 < param_array.size else 0.0
    c = param_array[idx_plane + 2] if idx_plane + 2 < param_array.size else 0.0

    for k in range(x.size):
        z.flat[k] += a * x.flat[k] + b * y.flat[k] + c

    return z

def build_sum_gaussians_wrapper(n):
    def model_func(x, y, **params):
        param_array = np.empty(n*6 + 3, dtype=np.float64)
        pos = 0

        for i in range(n):
            param_array[pos]   = params[f'g{i}_amplitude']
            param_array[pos+1] = params[f'g{i}_radius']
            param_array[pos+2] = params[f'g{i}_angle']
            param_array[pos+3] = params[f'g{i}_radius_width']
            param_array[pos+4] = params[f'g{i}_angle_width']
            param_array[pos+5] = params[f'g{i}_theta']
            pos += 6

        param_array[pos]   = params['A']
        param_array[pos+1] = params['B']
        param_array[pos+2] = params['C']

        return sum_of_gaussians_and_plane(x, y, param_array, n)
    return model_func

def compute_initial_params(sub, x0, y0, x1, y1, debug = False):
    """Compute initial 2D Gaussian parameters from subarray."""
    amp = np.nanpercentile(sub, 99)  # the literal max is a biased-high estimate of a noisy peak
    if debug:
        print("compute_initial_params")
    com_y, com_x = safe_center_of_mass(sub-np.nanpercentile(sub, 10))
    xo = x0 + com_x
    yo = y0 + com_y
    sigma_x = max((x1 - x0) / 2 / 2.355, 1.0)  # FWHM to sigma conversion
    sigma_y = max((y1 - y0) / 2 / 2.355, 1.0)
    return amp, xo, yo, sigma_x, sigma_y


def fit_peak_cluster(cluster, boxes, img, peaks_pool = None, theta_fixed=False, debug=False, hot_pixel_percentile=None):
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
    roi = np.array(img[ymin:ymax, xmin:xmax])
    mask = np.isfinite(roi)

    # Mask out every other detected box overlapping this ROI, vectorized over
    # all of `boxes` at once instead of a per-box Python loop.
    cluster_idx_set = set(cluster.indices.tolist())
    other_boxes = [b for b in boxes if b.index not in cluster_idx_set]
    if other_boxes:
        limits = np.array([b.limits for b in other_boxes])
        h_roi, w_roi = roi.shape
        rx0 = np.clip(np.round(limits[:, 0] - xmin), 0, w_roi).astype(int)
        rx1 = np.clip(np.round(limits[:, 2] - xmin), 0, w_roi).astype(int)
        ry0 = np.clip(np.round(limits[:, 1] - ymin), 0, h_roi).astype(int)
        ry1 = np.clip(np.round(limits[:, 3] - ymin), 0, h_roi).astype(int)
        for i in np.nonzero((rx1 > rx0) & (ry1 > ry0))[0]:
            mask[ry0[i]:ry1[i], rx0[i]:rx1[i]] = False

    # Hot pixels: this ROI's own percentile, folded into the same mask/roi
    # write as the other-box exclusion above.
    if hot_pixel_percentile is not None and roi.size > 0 and np.isfinite(roi).any():
        hot_thresh = np.nanpercentile(roi, hot_pixel_percentile)
        mask &= ~(roi > hot_thresh)

    roi[~mask] = np.nan


    for (mxmin, mymin, mxmax, mymax) in cluster.mask_boxes:
        rxmin = int(np.clip(mxmin - xmin, 0, roi.shape[1]))
        rxmax = int(np.clip(mxmax - xmin, 0, roi.shape[1]))
        rymin = int(np.clip(mymin - ymin, 0, roi.shape[0]))
        rymax = int(np.clip(mymax - ymin, 0, roi.shape[0]))
        mask[rymin:rymax, rxmin:rxmax] = False
    roi_bkg = np.array(roi)
    roi_bkg[~mask] = np.nan

    Y, X = np.mgrid[ymin:ymax, xmin:xmax]
    data = roi.ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    valid_mask = np.isfinite(data)
    data = data[valid_mask]
    X_flat = X_flat[valid_mask]
    Y_flat = Y_flat[valid_mask]

    # Plane params
    a_bkg, b_bkg, c_bkg = 0, 0, np.nanpercentile(roi_bkg, 10)
    X_plane, Y_plane = X, Y
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
        # go to the sample horizon
        if y0 < h/90*5:
            y0 = 0
            vary_y0 = False
        else:
            vary_y0 = True
        if debug:
            print("amp, xo, yo, sigma_x, sigma_y, vary_y0 ", amp, xo, yo, sigma_x, sigma_y, vary_y0)
        x_bound_min = np.clip(x0 - (x1 - x0)/4, xmin, xmax)
        x_bound_max = np.clip(x1 + (x1 - x0)/4 , xmin, xmax)
        y_bound_min = np.clip(y0 - (y1 - y0)/4 , ymin, ymax)
        y_bound_max = np.clip(y1 + (y1 - y0)/4 , ymin, ymax) if not box.is_cut_qz else h

        # if x_bound_min == x_bound_max:
        #     print(x_bound_min, x_bound_max, x0, x1, xmin, xmax)
        #     print(box.limits)
        # Add Gaussian parameters to the model
        params.add(f'g{i}_amplitude', value=amp, min=0)
        params.add(f'g{i}_radius', value=xo, min=x_bound_min, max=x_bound_max)
        params.add(f'g{i}_angle', value=yo, min=y_bound_min, max=y_bound_max, vary = vary_y0)
        params.add(f'g{i}_radius_width', value=sigma_x, min=(x1-x0)/8, max = (x1-x0)) #/2
        params.add(f'g{i}_angle_width', value=sigma_y, min=(y1-y0)/8, max = (y1-y0)) #/2
        params.add(f'g{i}_theta', value=0, vary=not theta_fixed)

    # Add parameters for the background plane
    params.add('A', value=a_bkg, min = -0.1, max = 0.1)
    params.add('B', value=b_bkg, min = -1, max = 1)
    params.add('C', value=c_bkg, min = -abs(c_bkg/4)-1, max = abs(c_bkg*2)+1)

    # Exit if no valid peaks
    if len(params) == 0:
        return None, None, None

    # An entirely-invalid background region (e.g. a detector gap in real
    # data) makes np.nanpercentile return NaN, which propagates into C's
    # bounds too -- sanitize both, not just values, or scipy rejects the
    # (now-valid) value as outside (still-NaN) bounds.
    for name, p in params.items():
        if np.isnan(p.value):
            p.value = 1.0 if "amplitude" in name else 0.0
        if p.min is not None and np.isnan(p.min):
            p.min = -np.inf
        if p.max is not None and np.isnan(p.max):
            p.max = np.inf

    time1 = time.time()

    n_peaks = len(cluster.indices)
    use_jac = _USE_ANALYTIC_JAC and n_peaks <= _MAX_ANALYTIC_JAC_COMPONENTS
    var_names = [name for name, p in params.items() if p.vary]

    model_func = model.func
    params_working = copy.deepcopy(params)

    def _residual(xvec):
        for name, v in zip(var_names, xvec):
            params_working[name].value = v
        kwargs = {name: p.value for name, p in params_working.items()}
        return data - model_func(X_flat, Y_flat, **kwargs)

    def _jac(xvec):
        for name, v in zip(var_names, xvec):
            params_working[name].value = v
        derivs = _gaussian2d_terms(params_working, X_flat, Y_flat, n_peaks)
        return _params_to_jacobian(params_working, derivs, None)

    # Fit directly via scipy (bypasses lmfit's own per-iteration bookkeeping)
    list_to_return = _fit_with_scipy(_residual, _jac, params, var_names, use_jac)
    time2 = time.time()

    if debug:
        try:
            for name, v in list_to_return['params'].items():
                params_working[name].value = v
            result = _DebugResultShim(params_working)
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
        except Exception as e:
            print(f"[debug plot failed for cluster {cluster.indices.tolist()}, skipping]: {e}")

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
    if not np.any(roi > 0):
        return
    norm = LogNorm(vmin=np.nanmin(roi[roi > 0]), vmax=np.nanmax(roi))
    fig, axes = plt.subplots(figsize=(6, 6))
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
            edgecolor='blue',
            facecolor='none',
            linewidth=2,
            alpha=1,
            linestyle='--',
            label=f'init peak {idx}'
        )
        theta = params.get(f'g{i}_theta', None)
        if theta is not None:
            ellipse.angle = np.degrees(-theta.value)
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
            angle=np.degrees(-theta.value),
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

        cos_t = cos(theta)
        sin_t = sin(theta)
        sin2 = 2.0 * sin_t * cos_t

        a = (cos_t * cos_t) / (2.0 * sigx * sigx) + (sin_t * sin_t) / (2.0 * sigy * sigy)
        b = -sin2 / (4.0 * sigx * sigx) + sin2 / (4.0 * sigy * sigy)
        c = (sin_t * sin_t) / (2.0 * sigx * sigx) + (cos_t * cos_t) / (2.0 * sigy * sigy)

        dx = x - xo
        dy = y - yo
        z += ne.evaluate("amp * exp(-(a*dx*dx + 2*b*dx*dy + c*dy*dy))")

    # Plane background
    offset_plane = n * 6
    a_plane = param_array[offset_plane]
    b_plane = param_array[offset_plane + 1]
    c_plane = param_array[offset_plane + 2]

    z += ne.evaluate("a_plane * x + b_plane * y + c_plane")

    # 1D Gaussians along x
    offset_1d = offset_plane + 3
    for j in range(m):
        base = offset_1d + j * 3
        amp_1d = param_array[base]
        center_1d = param_array[base + 1]
        sigma_1d = param_array[base + 2]

        dx = x - center_1d
        inv2sigma2 = 0.5 / (sigma_1d * sigma_1d)
        z += ne.evaluate("amp_1d * exp(-dx*dx*inv2sigma2)")

    return z

def gaussian_height(x, radius, amplitude, radius_width):
    dx = x - radius
    inv2sigma2 = 0.5 / (radius_width * radius_width)
    return ne.evaluate("amplitude * exp(-dx*dx*inv2sigma2)")


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



def fit_peak_on_ring_cluster(cluster, boxes, img, peaks_pool, theta_fixed, debug = False, hot_pixel_percentile=None):
    time0 = time.time()
    xmin, ymin, xmax, ymax = np.round(cluster.bbox).astype(int)
    h, w = img.shape
    xmin = np.clip(xmin, 0, w)
    xmax = np.clip(xmax, 0, w)
    ymin = np.clip(ymin, 0, h)
    ymax = np.clip(ymax, 0, h)

    # Extract ROI from the image
    roi = np.array(img[ymin:ymax, xmin:xmax])
    y_len, x_len = roi.shape

    # Mask out every other detected box overlapping this ROI, same as
    # fit_peak_cluster -- otherwise a wide ring's bbox can pull in unrelated
    # peaks' pixels uncorrected into this cluster's background and fit data.
    mask = np.isfinite(roi)
    cluster_idx_set = set(cluster.indices.tolist())
    other_boxes = [b for b in boxes if b.index not in cluster_idx_set]
    if other_boxes:
        limits = np.array([b.limits for b in other_boxes])
        h_roi, w_roi = roi.shape
        rx0 = np.clip(np.round(limits[:, 0] - xmin), 0, w_roi).astype(int)
        rx1 = np.clip(np.round(limits[:, 2] - xmin), 0, w_roi).astype(int)
        ry0 = np.clip(np.round(limits[:, 1] - ymin), 0, h_roi).astype(int)
        ry1 = np.clip(np.round(limits[:, 3] - ymin), 0, h_roi).astype(int)
        for i in np.nonzero((rx1 > rx0) & (ry1 > ry0))[0]:
            mask[ry0[i]:ry1[i], rx0[i]:rx1[i]] = False

    # Hot pixels: this ROI's own percentile, folded into the same mask/roi
    # write as the other-box exclusion above.
    if hot_pixel_percentile is not None and roi.size > 0 and np.isfinite(roi).any():
        hot_thresh = np.nanpercentile(roi, hot_pixel_percentile)
        mask &= ~(roi > hot_thresh)

    roi[~mask] = np.nan

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
            amp = np.nanpercentile(sub, 99)
            yy, xx = np.indices(sub.shape)
            com_y, com_x = safe_center_of_mass(sub)
            xo = x0 + com_x
            yo = y0 + com_y
            sigma_x = max((x1 - x0) /2/2.355, 1.0)  # FWHM to sigma conversion
            sigma_y = max((y1 - y0) /2/2.355, 1.0)

        # go to the sample horizon
        if y0 < h/90*5:
            y0 = 0
            vary_y0 = False
        else:
            vary_y0 = True
        if debug:
            print("amp, xo, yo, sigma_x, sigma_y, vary_y0 ", amp, xo, yo, sigma_x, sigma_y, vary_y0)

        x_bound_min = np.clip(x0 - (x1 - x0)/4, xmin, xmax)
        x_bound_max = np.clip(x1 + (x1 - x0)/4 , xmin, xmax)
        y_bound_min = np.clip(y0 - (y1 - y0)/4 , ymin, ymax)
        y_bound_max = np.clip(y1 + (y1 - y0)/4 , ymin, ymax) if not box.is_cut_qz else h

        # Add Gaussian parameters to the model
        params.add(f'g{i}_amplitude', value=amp, min=0)
        params.add(f'g{i}_radius', value=xo, min=x_bound_min, max=x_bound_max)
        params.add(f'g{i}_angle', value=yo, min=y_bound_min, max=y_bound_max, vary = vary_y0)
        params.add(f'g{i}_radius_width', value=sigma_x, min=(x1 - x0) / 8, max=(x1 - x0)) # / 2
        params.add(f'g{i}_angle_width', value=sigma_y, min=(y1 - y0) / 8, max=(y1 - y0)) # / 2
        params.add(f'g{i}_theta', value=0, vary=not theta_fixed)

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
        if p.min is not None and np.isnan(p.min):
            p.min = -np.inf
        if p.max is not None and np.isnan(p.max):
            p.max = np.inf

    if debug:
        try:
            debug_params_out_of_bounds(params, boxes, peak_indices, ring_indices, roi, cluster)
        except Exception as e:
            print(f"[debug_params_out_of_bounds failed for cluster {cluster.indices.tolist()}, skipping]: {e}")

    n_peaks, n_rings = len(peak_indices), len(ring_indices)
    use_jac = _USE_ANALYTIC_JAC and (n_peaks + n_rings) <= _MAX_ANALYTIC_JAC_COMPONENTS
    var_names = [name for name, p in params.items() if p.vary]

    model_func = model.func
    params_working = copy.deepcopy(params)

    def _residual(xvec):
        for name, v in zip(var_names, xvec):
            params_working[name].value = v
        kwargs = {name: p.value for name, p in params_working.items()}
        return data - model_func(X_flat, Y_flat, **kwargs)

    def _jac(xvec):
        for name, v in zip(var_names, xvec):
            params_working[name].value = v
        derivs = _gaussian2d_terms(params_working, X_flat, Y_flat, n_peaks)
        derivs.update(_gaussian1d_terms(params_working, X_flat, n_rings, prefix='g1d_'))
        return _params_to_jacobian(params_working, derivs, None)

    # A size-based "prefer lsmr for n_components>=4" rule was tried here, but
    # it was tuned against clusters whose ROI leaked unrelated boxes' pixels
    # into the fit (see the masking added above). With that fixed, the rule
    # no longer reliably predicts anything -- it can make lsmr fail outright
    # where plain 'exact' already succeeds quickly. Back to unconditional
    # 'exact' first, 'lsmr' only as a retry on an actual failure.
    list_to_return = _fit_with_scipy(_residual, _jac, params, var_names, use_jac)

    time2 = time.time()

    if debug:
        try:
            for name, v in list_to_return['params'].items():
                params_working[name].value = v
            result = _DebugResultShim(params_working)
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
        except Exception as e:
            print(f"[debug plot failed for cluster {cluster.indices.tolist()}, skipping]: {e}")

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

        x_bound_min = np.clip(x0_box - (x1_box - x0_box) / 4, xmin, xmax)
        x_bound_max = np.clip(x0_box + (x1_box - x0_box) / 4, xmin, xmax)

        gparams[f'g{i}_amplitude'].min = 0
        gparams[f'g{i}_radius_width'].min = 0
        gparams[f'g{i}_radius'].min = x_bound_min
        gparams[f'g{i}_radius_width'].max = x1_box - x0_box

        params.update(gparams)

    n_rings_1d = len(cluster.indices)
    use_jac = _USE_ANALYTIC_JAC and n_rings_1d <= _MAX_ANALYTIC_JAC_COMPONENTS
    var_names = [name for name, p in params.items() if p.vary]
    params_working = copy.deepcopy(params)

    # Same math the composite LinearModel + sum(Model(gaussian_height)) evaluates
    # to, computed directly (no lmfit composite-model overhead).
    def _ring_model_at(p, xdata):
        z = p['lin_slope'].value * xdata + p['lin_intercept'].value
        for i in range(n_rings_1d):
            z = z + gaussian_height(xdata, p[f'g{i}_radius'].value,
                                    p[f'g{i}_amplitude'].value, p[f'g{i}_radius_width'].value)
        return z

    def _residual(xvec):
        for name, v in zip(var_names, xvec):
            params_working[name].value = v
        return profile - _ring_model_at(params_working, x)

    def _jac(xvec):
        for name, v in zip(var_names, xvec):
            params_working[name].value = v
        derivs = _gaussian1d_terms(params_working, x, n_rings_1d, prefix='g')
        derivs['lin_slope'] = x
        derivs['lin_intercept'] = np.ones_like(x)
        return _params_to_jacobian(params_working, derivs, None)

    mask = np.isfinite(profile)
    profile = profile[mask]
    x = x[mask]
    result = _fit_with_scipy(_residual, _jac, params, var_names, use_jac)

    if debug:
        try:
            for name, v in result['params'].items():
                params_working[name].value = v
            plt.figure(figsize=(6, 4))
            plt.plot(x, profile, 'b', label='Data')
            plt.plot(x, _ring_model_at(params_working, x), 'r-', label='Best Fit')
            plt.plot(x, _ring_model_at(params, x), 'c--', label='Initial Guess')
            plt.title(f"Cluster {cluster.indices.tolist()}")
            plt.xlabel('X [pixels]')
            plt.ylabel('Mean intensity')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[debug plot failed for cluster {cluster.indices.tolist()}, skipping]: {e}")

    return result



def process_cluster_args(args):
    cluster, cluster_type, boxes, img, masked_img, theta_fixed, debug = args
    if cluster_type == 'rings':
        result = fit_ring_cluster(cluster, boxes, masked_img, None, debug)
    elif cluster_type == 'peaks':
        result = fit_peak_cluster(cluster, boxes, img, None, theta_fixed, debug)
    elif cluster_type == 'both':
        result = fit_peak_on_ring_cluster(cluster, boxes, img, None, theta_fixed, debug)
    else:
        result = None
    return cluster, result

##### MP with shared memory
def fit_clusters_multiprocessing(clusters, boxes, img, masked_img, theta_fixed, debug=False):
    cluster_types = ['rings', 'peaks', 'both']

    shm_img, shm_masked, img_shape, masked_shape, img_dtype, masked_dtype = init_shared_images(img, masked_img)

    try:
        for ctype in cluster_types:
            ctype_clusters = [(cluster, ctype, boxes,
                               shm_img.name, shm_masked.name,
                               img_shape, masked_shape, img_dtype, masked_dtype, theta_fixed,
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
    cluster, ctype, boxes, shm_img_name, shm_masked_name, img_shape, masked_shape, img_dtype, masked_dtype, theta_fixed, debug = args

    existing_shm_img = shared_memory.SharedMemory(name=shm_img_name)
    existing_shm_masked = shared_memory.SharedMemory(name=shm_masked_name)
    img = np.ndarray(img_shape, dtype=img_dtype, buffer=existing_shm_img.buf)
    masked_img = np.ndarray(masked_shape, dtype=masked_dtype, buffer=existing_shm_masked.buf)

    fitting_result = process_cluster_args((cluster, ctype, boxes, img, masked_img, theta_fixed, debug))

    return fitting_result

def debug_params_out_of_bounds(params, boxes, peak_indices, ring_indices, roi, cluster):
    import re

    # --- ROI check ---
    if roi.size == 0:
        print("\n--- ROI ISSUE ---\nROI is empty\n-----------------\n")
        print("cluster.bbox", cluster.bbox)
        print("cluster", cluster)
    elif not np.isfinite(roi).any():
        print("\n--- ROI ISSUE ---\nROI contains only NaNs\n----------------------\n")
        print("cluster.bbox", cluster.bbox)
        print("cluster", cluster)

    for name, p in params.items():
        if p.min is None or p.max is None:
            continue

        invalid = np.isnan(p.min) or np.isnan(p.max)
        out = not (p.min <= p.value <= p.max)

        if not (invalid or out):
            continue

        print(f"\n--- PARAMETER ISSUE ---\n{name}: {p.value} [{p.min}, {p.max}]")
        print("reason :", "NaN bounds" if invalid else "out of bounds")

        # --- resolve box ---
        idx = None
        is_peak = False

        m = re.search(r'g(\d+)_', name)
        if m:
            idx = int(m.group(1))
            if idx < len(peak_indices):
                idx = peak_indices[idx]
                is_peak = True

        m = re.search(r'g1d_(\d+)_', name)
        if m:
            i = int(m.group(1))
            if i < len(ring_indices):
                idx = ring_indices[i]

        if idx is not None:
            box = boxes[idx]
            x0, y0, x1, y1 = map(lambda v: int(np.round(v)), box.limits)

            print(f"box_idx: {idx}, limits: {box.limits}")
            print(f"size   : ({x1-x0}, {y1-y0})")

            if is_peak:
                print(f"is_cut_qz: {getattr(box, 'is_cut_qz', 'N/A')}")

        print("--------------------------------")