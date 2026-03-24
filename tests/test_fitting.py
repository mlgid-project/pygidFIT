import pytest
import numpy as np
from pygidfit import ProcessDataFromFile
from pygid import NexusFile


@pytest.mark.parametrize(
    "filename, entry, frame_num, theta_fixed",
    [
        ('./example/251210_HATCN_labeled.h5', None, None, True),  # Rings
        ('./example/251210_DIP_labeled.h5', None, None, True),    # Peaks
        ('./example/BA2PbI4.h5', None, None, True),               # Both
        ('./example/BA2PbI4.h5', 'entry_0001', None, True),       # Single entry
        ('./example/BA2PbI4.h5', 'entry_0000', 1, True),          # Single frame
        ('./example/251210_DIP_labeled.h5', None, None, False),   # Unfixed theta
    ]
)
def test_process_data(filename, entry, frame_num, theta_fixed):
    """
    Integration test for ProcessDataFromFile on different datasets.
    Checks that the class runs without error and outputs expected datasets.
    """
    analysis = ProcessDataFromFile(
        filename,
        entry=entry,
        frame_num=frame_num,
        crit_angle=2,
        clustering_distance_rings=10,
        clustering_distance_peaks=10,
        clustering_extend=2,
        use_pool=False,
        debug=False,
        theta_fixed=theta_fixed,
    )

    # Verify entry_dict exists
    assert hasattr(analysis, 'entry_dict'), "ProcessDataFromFile missing entry_dict attribute"

    nexus = NexusFile(filename)

    entries_to_check = [entry] if entry else list(analysis.entry_dict.keys())

    for e in entries_to_check:
        frames_to_check = [frame_num] if frame_num is not None else range(analysis.entry_dict[e]['shape'][0])
        for f in frames_to_check:
            _check_single_image(nexus, e, f, theta_fixed)


def _check_single_image(nexus, entry, frame, theta_fixed):
    """
    Helper function to check a single frame's fitted peaks.
    """
    # Check required datasets exist
    fitted_peaks = nexus.get_dataset(f'{entry}/data/analysis/frame{frame:05d}/fitted_peaks')
    fitted_peaks_errors = nexus.get_dataset(f'{entry}/data/analysis/frame{frame:05d}/fitted_peaks_errors')
    assert fitted_peaks is not None, f"Fitted peaks missing for {entry} frame {frame}"
    assert fitted_peaks_errors is not None, f"Fitted peaks errors missing for {entry} frame {frame}"

    # Validate lengths match detected peaks
    detected_peaks = nexus.get_dataset(f'{entry}/data/analysis/frame{frame:05d}/detected_peaks')
    assert len(detected_peaks['amplitude']) == len(fitted_peaks['amplitude']), \
        f"Number of fitted peaks does not match detected peaks for {entry} frame {frame}"
    assert len(detected_peaks['amplitude']) == len(fitted_peaks_errors['amplitude']), \
        f"Number of fitted peak errors does not match detected peaks for {entry} frame {frame}"

    # Validate theta values
    if np.isfinite(fitted_peaks['theta']).any():
        if theta_fixed:
            assert np.nanmax(fitted_peaks['theta']) == 0, f"Theta is not fixed for {entry} frame {frame}"
        else:
            assert np.nanmax(fitted_peaks['theta']) != 0, f"Theta should not be fixed for {entry} frame {frame}"

    # Cleanup datasets to avoid side effects
    nexus.delete_dataset(f'{entry}/data/analysis/frame{frame:05d}/fitted_peaks')
    nexus.delete_dataset(f'{entry}/data/analysis/frame{frame:05d}/fitted_peaks_errors')