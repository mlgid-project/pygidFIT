import h5py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


pygid_results_dtype = np.dtype([
        ('amplitude', 'f4'),
        ('angle', 'f4'),
        ('angle_width', 'f4'),
        ('radius', 'f4'),
        ('radius_width', 'f4'),
        ('q_xy', 'f4'),
        ('q_z', 'f4'),
        ('theta', 'f4'),
        ('score', 'f4'),
        ('A', 'f4'),
        ('B', 'f4'),
        ('C', 'f4'),
        ('is_ring', 'bool'),
        ('is_cut_qz', 'bool'),
        ('is_cut_qxy', 'bool'),
        ('visibility', 'i4'),
        ('id', 'i4'),
    ])

@dataclass
class DetectedPeaks:
    analysis: h5py.Group = None
    frame_num: int = None
    frame_key: str = field(init=False)
    data: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.analysis is not None:
            self.frame_key = list(self.analysis.keys())[self.frame_num]
            frame_group = self.analysis[self.frame_key]
            if "detected_peaks" in frame_group:
                ds = frame_group["detected_peaks"]
                names = ds.dtype.names
                self.data = {name: ds[name] for name in names}
                self.__dict__.update(self.data)
            else:
                raise ValueError("No detected peaks found")
        # for key in frame_group['detected_peaks'].keys():
        #     self.data[key] = frame_group[f'detected_peaks/{key}'][()]
        # self.__dict__.update(self.data)

@dataclass
class DataBatch:
    detected_peaks: list = None
    raw_giwaxs: np.ndarray = None
    ai: list = None
    wavelength: float = None
    q_xy: np.ndarray = None
    q_z: np.ndarray = None

@dataclass
class DataSaver:
    img_container_list: list = None
    filename: str = None
    entry: str = None
    batch_num: int = 0
    batch_size: int = 10

    def __post_init__(self):
        with h5py.File(self.filename, 'r+') as f:
            for i in range(len(self.img_container_list)):
                analysis = f[f"{self.entry}/data/analysis"]
                folder_num = i+self.batch_num*self.batch_size
                results_array = get_results_array(self.img_container_list[i])
                results_err_array = get_results_err_array(self.img_container_list[i])
                group = analysis[list(analysis.keys())[folder_num]]
                if 'fitted_peaks' in group:
                    del group['fitted_peaks']
                group.create_dataset('fitted_peaks', data=results_array, dtype=pygid_results_dtype)
                if 'fitted_peaks_errors' in group:
                    del group['fitted_peaks_errors']
                group.create_dataset('fitted_peaks_errors', data=results_err_array, dtype=pygid_results_dtype)



def get_results_array(img_container):
    results_array = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
    results_array['amplitude'] = img_container.amplitude
    results_array['angle'] = img_container.angle
    results_array['angle_width'] = img_container.angle_width
    results_array['radius'] = img_container.radius
    results_array['radius_width'] = img_container.radius_width
    results_array['q_z'] = img_container.qzqxyboxes[0]
    results_array['q_xy'] = img_container.qzqxyboxes[1]
    results_array['theta'] = img_container.theta
    results_array['A'] = img_container.A
    results_array['B'] = img_container.B
    results_array['C'] = img_container.C
    results_array['is_ring'] = img_container.is_ring
    results_array['is_cut_qz'] = img_container.is_cut_qz
    results_array['is_cut_qxy'] = img_container.is_cut_qxy
    results_array['visibility'] = img_container.visibility
    results_array['score'] = img_container.score
    results_array['id'] = img_container.id
    return results_array

def get_results_err_array(img_container):
    results_array = np.zeros(len(img_container.radius_width), dtype=pygid_results_dtype)
    results_array['amplitude'] = img_container.amplitude_err
    results_array['angle'] = img_container.angle_err
    results_array['angle_width'] = img_container.angle_width_err
    results_array['radius'] = img_container.radius_err
    results_array['radius_width'] = img_container.radius_width_err
    results_array['q_z'] = img_container.qzqxyboxes_err[0]
    results_array['q_xy'] = img_container.qzqxyboxes_err[1]
    results_array['theta'] = img_container.theta_err
    results_array['A'] = img_container.A_err
    results_array['B'] = img_container.B_err
    results_array['C'] = img_container.C_err
    results_array['is_ring'] = img_container.is_ring
    results_array['is_cut_qz'] = img_container.is_cut_qz
    results_array['is_cut_qxy'] = img_container.is_cut_qxy
    results_array['visibility'] = img_container.visibility
    results_array['score'] = img_container.score
    results_array['id'] = img_container.id
    return results_array



@dataclass
class DataLoader:
    filename: str
    entry_list: Optional[List[str]] = None
    entry_num: int = 0
    batch_size: int = 10
    batch_num: int = 0
    data: DataBatch = None
    entry_done: bool = False
    debug: bool = False

    def __post_init__(self):
        if self.entry_list is None:
            self.entry_list = self._load_file_structure()
            # if len(self.entry_list) == 1:
            #     return self.load_entry(self.entry_list[0]), self.entry_list
            # else:
            return None, self.entry_list
        else:
            return self.load_entry(self.entry_list[self.entry_num]), self.entry_list

    def load_entry(self, entry: str):
        with h5py.File(self.filename, 'r') as f:
            analysis = f[f"{entry}/data/analysis"]
            folder_num = len(analysis.keys())
            ind1 = self.batch_num * self.batch_size
            ind2 = min((self.batch_num + 1) * self.batch_size, folder_num)
            if self.debug:
                print("Loading frame number: from ", ind1, " to ",ind2)
            if ind2 == folder_num:
                self.entry_done = True
            frame_nums = np.arange(ind1, ind2)
            self.data = DataBatch()
            self.data.entry = entry
            self.data.frame_num = frame_nums
            self.data.detected_peaks = self.load_detected_peaks(analysis, frame_nums)
            self.data.raw_giwaxs = self.load_image(f, entry, ind1, ind2)
            self.data.rad_max_px = np.sqrt(self.data.raw_giwaxs.shape[1]**2 + self.data.raw_giwaxs.shape[2]**2)

            ai, wavelength, q_xy, q_z = self.load_metadata(f, entry, ind1, ind2)
            self.data.ai = ai
            self.data.wavelength = wavelength
            self.data.q_xy = q_xy
            self.data.q_z = q_z
            self.data.q_abs_max = np.sqrt(np.nanmax(q_xy**2) + np.nanmax(q_z**2))
        return self.data

    def load_detected_peaks(self, analysis, frame_num_list: np.ndarray):
        detected_peaks = []
        for frame_num in frame_num_list:
            detected_peaks.append(DetectedPeaks(analysis, frame_num))
        return detected_peaks
    # def load_detected_peaks(self, analysis, frame_num_list: np.ndarray):
    #     detected_peaks = []
    #     for frame_num in frame_num_list:
    #         detected_peaks.append(DetectedPeaks(analysis, frame_num))
    #     return detected_peaks

    def load_image(self, f, entry: str, ind1: int, ind2: int):
        return f[f"{entry}/data/img_gid_q"][ind1:ind2].astype('float32')

    def load_metadata(self, f, entry: str, ind1: int, ind2: int):
        try:
            ai = f[f"{entry}/instrument/angle_of_incidence"][ind1:ind2]
        except:
            ai = [f[f"{entry}/instrument/angle_of_incidence"][()]]
        wavelength = f[f"{entry}/instrument/monochromator/wavelength"][()] * 1e10
        q_xy = f[f"{entry}/data/q_xy"][()].astype('float32')
        q_z = f[f"{entry}/data/q_z"][()].astype('float32')
        return np.array(ai), wavelength, q_xy, q_z

    def _load_file_structure(self) -> List[str]:
        with h5py.File(self.filename, 'r') as f:
            entry_list = []
            for key in f.keys():
                group = f[f"{key}/data"]
                if "img_gid_q" in group:
                    entry_list.append(key)
            if len(entry_list) == 0:
                raise ValueError("No entries found in the HDF5 file.")
            return entry_list
