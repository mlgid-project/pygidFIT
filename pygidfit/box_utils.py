import numpy as np
from dataclasses import dataclass

@dataclass
class Boxes:
    limits: np.ndarray
    is_ring: bool
    index: int
    # cluster_num: List[int] = field(default_factory=list)
    is_cut_qz: bool = False
    is_cut_qxy: bool = False
    fitting_result = None


def find_box_type(is_cut_qxy, is_cut_qz, box, ratio_threshold = 10) -> bool:
    """
    Classifies a bounding box as either a 'line' or a 'peak' based on its aspect ratio.

    Args:
        box (array-like): [xmin, ymin, xmax, ymax]
        ratio_threshold (float): threshold above which a box is considered a line

    Returns:
        str: 'line' or 'peak'
    """
    result1 = True if (is_cut_qz and is_cut_qxy) else False

    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    aspect_ratio = max(height / width, width / height)

    result2 =  True if aspect_ratio >= ratio_threshold else False
    return result1 or result2

    #
    # return True if shape <= height*1.5 else False




def boxes_preprocessing(detected_peaks, polar_shape, wavelength, q_abs_max):

    radius1_q = detected_peaks.radius - (detected_peaks.radius_width / 2)
    radius2_q = detected_peaks.radius + (detected_peaks.radius_width / 2)


    radius1 = np.round(radius1_q / q_abs_max * polar_shape[1]) - 1
    radius2 = np.round(radius2_q / q_abs_max * polar_shape[1]) + 1

    theta1_deg = detected_peaks.angle - (detected_peaks.angle_width / 2)
    theta2_deg = detected_peaks.angle + (detected_peaks.angle_width / 2)
    theta1 = np.round(theta1_deg / 90 * polar_shape[0]).astype(int) - 1
    theta2 = np.round(theta2_deg / 90 * polar_shape[0]).astype(int) + 1

    # to make boxes inside the img
    radius1 = np.clip(radius1, 0, polar_shape[1])
    radius2 = np.clip(radius2, 0, polar_shape[1])
    theta1 = np.clip(theta1, 0, polar_shape[0])
    theta2 = np.clip(theta2, 0, polar_shape[0])


    boxes = np.stack([radius1,
                           theta1,
                           radius2,
                           theta2], axis=1)
    boxes_q_deg = np.stack([radius1_q,
                           theta1_deg,
                           radius2_q,
                           theta2_deg], axis=1)
    boxes_list = []
    for i in range(len(radius1)):
        is_cut_qxy, is_cut_qz = _get_cut_flags(radius1_q[i],theta1_deg[i],radius2_q[i],theta2_deg[i],wavelength)
        boxes_list.append(Boxes(boxes[i], find_box_type(is_cut_qxy, is_cut_qz, boxes[i]), i,is_cut_qxy, is_cut_qz ))
        boxes_list[i].boxes_q_deg = boxes_q_deg[i]
    return boxes_list

def _get_cut_flags(radius1, theta1_deg, radius2, theta2_deg, wavelength):
    if theta1_deg < 2:
        is_cut_qxy = True
    else:
        is_cut_qxy = False
    is_cut_qz = _get_missing_wedge_pol(wavelength, radius2, theta2_deg)
    return is_cut_qz, is_cut_qxy

def _get_missing_wedge_pol(wavelength, q_abs, phi):
    k = 2 * np.pi / float(wavelength)
    return bool(np.abs(q_abs) > np.abs(2*k*np.cos(np.deg2rad(phi))))


def make_box_attributes(indices, boxes, fitting_result, type = None, debag = False):

    params = fitting_result['params']
    errors = fitting_result['errors']

    i = 0
    for ind in indices:
        box = boxes[ind]
        if box.is_ring and box.fitting_result is not None:
            continue
        box.fitting_result = {}
        for key in params.keys():
            if key.startswith(f'g{i}_'):
                key_changed = key.replace(f'g{i}_', '')
                box.fitting_result[key_changed] = params[key]
            elif key in ['A', 'B', 'C']:
                box.fitting_result[key] = params[key]
            elif key == 'lin_slope':
                box.fitting_result['A'] = params[key]
            elif key == 'lin_intercept':
                box.fitting_result['C'] = params[key]

        if not 'B' in box.fitting_result:
            box.fitting_result['B'] = 0
        if not 'angle' in box.fitting_result:
            box.fitting_result['angle'] = np.nan
        if not 'angle_width' in box.fitting_result:
            box.fitting_result['angle_width'] = np.inf
        if not 'theta' in box.fitting_result:
            box.fitting_result['theta'] = np.nan

        box.fitting_error = {}
        for key in errors.keys():
            if key.startswith(f'g{i}_'):
                key_changed = key.replace(f'g{i}_', '')
                box.fitting_error[key_changed] = errors[key]
            elif key in ['A', 'B', 'C']:
                box.fitting_error[key] = errors[key]
            elif key == 'lin_slope':
                box.fitting_error['A'] = errors[key]
            elif key == 'lin_intercept':
                box.fitting_error['C'] = errors[key]

        if not 'B' in box.fitting_error:
            box.fitting_error['B'] = 0
        if not 'angle' in box.fitting_error:
            box.fitting_error['angle'] = np.nan
        if not 'angle_width' in box.fitting_error:
            box.fitting_error['angle_width'] = np.inf
        if not 'theta' in box.fitting_error:
            box.fitting_error['theta'] = np.nan
        i+=1
        if debag:
            print("fitting_result",box.fitting_result)
            print("fitting_error", box.fitting_error)
