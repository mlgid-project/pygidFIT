from h5py import File
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import pygid
from typing import Any
from dataclasses import dataclass
def save_fit(filename, entry, img_container_fit, frame_num):
    with File(filename, "r+") as f:
        group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
        pygid._save_img_container_fit(f, group_name, img_container_fit)
        logger.info(f"Saved fitted peaks to file: {filename}, entry: {entry}, frame: {frame_num}")
        return


@dataclass
class DetectedPeaks:
    radius: Any = None
    radius_width: Any = None
    angle: Any = None
    angle_width: Any = None

    def __post_init__(self):
        pass

def read_detected_peaks(nexus, entry, frame_num):
    group_name = f"/{entry}/data/analysis/frame{str(frame_num).zfill(5)}"
    entry_dict = nexus.entry_dict
    if not entry in entry_dict:
        raise KeyError(f"entry {entry} not in the file. The file structure: {entry_dict}")
    return nexus.get_dataset(f"{group_name}/detected_peaks")

