import time
from pathlib import Path
import numpy as np

from pygidfit.process_scans import process_data_from_file


def run_scans(
        filename = None,
        batch_size = 10,
        crit_angle = 0,
        polar_shape = np.array([512,1024]),
        use_pool = False,
        debug = False,
        multiprocessing = True,
    ):
    """
    Main function to process a range of scans.
    
    Args:
        filename: Name of the file
        scan_range: Range of scan numbers to process (start, end)
        box_shape: Shape of the box (width, height)
        n_jobs: Number of parallel jobs to use (-1 for all cores)
    """

    process_data_from_file(
        filename,
        batch_size,
        crit_angle,
        polar_shape,
        use_pool,
        debug,
        multiprocessing
    )



# filename = 'S124_FAI_A1_after_gui_copy_with_duplicate.h5'
if __name__ == "__main__":
    filename = r'/tests/240305_PEN_DIP_0001.h5'
    run_scans(filename,
              polar_shape=[512 * 2, 1024 * 2],
              batch_size=1,
              crit_angle=0.1,
              debug=False,
              multiprocessing=False)
