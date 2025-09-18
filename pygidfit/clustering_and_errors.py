import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Cluster:
    bbox: np.ndarray      # [xmin, ymin, xmax, ymax] — bounding box of the cluster
    indices: np.ndarray   # (M,) — indices of the original boxes that belong to this cluster
    type: str = None


from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np


def cluster_peaks(peak_indices, ring_indices, centers, boxes, is_ring_list, r, extend):
    final_clusters = []

    if len(peak_indices) == 0:
        return []

    peak_centers = centers[peak_indices]
    tree_peaks = cKDTree(peak_centers)

    pairs = tree_peaks.query_pairs(r=r, p=np.inf)
    if pairs:
        i, j = zip(*pairs)
        row = np.concatenate([i, j])
        col = np.concatenate([j, i])
        data = np.ones(len(row), dtype=bool)
        adj = coo_matrix((data, (row, col)), shape=(len(peak_indices), len(peak_indices)))
        _, peak_labels = connected_components(adj, directed=False)
    else:
        peak_labels = np.arange(len(peak_indices))

    for cluster_id in np.unique(peak_labels):
        local_peak_idx = np.where(peak_labels == cluster_id)[0]
        cluster_peak_indices = peak_indices[local_peak_idx]
        cluster_boxes = boxes[cluster_peak_indices]
        cluster_centers = centers[cluster_peak_indices]

        xmin = np.min(cluster_boxes[:, 0]) - extend
        ymin = np.min(cluster_boxes[:, 1]) - extend
        xmax = np.max(cluster_boxes[:, 2]) + extend
        ymax = np.max(cluster_boxes[:, 3]) + extend

        h_box = min((ymax - ymin) * 0.5, r * 0.3)
        w_box = min((xmax - xmin), r * 0.3)
        xmin -= w_box
        ymin -= h_box
        xmax += w_box
        ymax += h_box

        bbox = np.array([xmin, ymin, xmax, ymax])

        cluster_ring_indices = []
        if len(ring_indices) > 0:
            cluster_x = centers[cluster_peak_indices][:, 0]
            for i in ring_indices:
                ring_x = centers[i, 0]
                if np.any(np.abs(cluster_x - ring_x) <= r):
                    cluster_ring_indices.append(i)

        cluster_ring_indices = np.array(cluster_ring_indices, dtype=int)
        all_indices = np.concatenate([cluster_peak_indices, cluster_ring_indices])
        all_is_ring = is_ring_list[all_indices]

        final_clusters.append(Cluster(
            bbox=bbox,
            indices=all_indices,
            type='both' if True in all_is_ring else 'peaks'
        ))

    return final_clusters


def cluster_rings(ring_indices, centers, boxes, r, extend):
    final_clusters = []

    if len(ring_indices) == 0:
        return []

    ring_centers = centers[ring_indices]
    x_coords = ring_centers[:, 0].reshape(-1, 1)

    tree = cKDTree(x_coords)
    pairs = tree.query_pairs(r=r, p=np.inf)

    if pairs:
        i, j = zip(*pairs)
        row = np.concatenate([i, j])
        col = np.concatenate([j, i])
        data = np.ones(len(row), dtype=bool)
        adj = coo_matrix((data, (row, col)), shape=(len(ring_indices), len(ring_indices)))
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(len(ring_indices))

    for cluster_id in np.unique(labels):
        local_idx = np.where(labels == cluster_id)[0]
        cluster_ring_indices = np.array(ring_indices)[local_idx]
        cluster_boxes = boxes[cluster_ring_indices]

        xmin = np.min(cluster_boxes[:, 0]) - extend * 5
        ymin = np.min(cluster_boxes[:, 1]) - extend * 5
        xmax = np.max(cluster_boxes[:, 2]) + extend * 5
        ymax = np.max(cluster_boxes[:, 3]) + extend * 5

        w_box = min((xmax - xmin), r * 0.2)
        xmin -= w_box
        xmax += w_box

        bbox = np.array([xmin, ymin, xmax, ymax])

        final_clusters.append(Cluster(
            bbox=bbox,
            indices=cluster_ring_indices,
            type='rings'
        ))

    return final_clusters


def cluster_boxes_by_centers(boxes_list, r, extend):
    boxes_id = np.array([box.index for box in boxes_list])
    is_ring_list = np.array([box.is_ring for box in boxes_list], dtype=bool)

    boxes = np.stack([box.limits for box in boxes_list])
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])

    peak_indices = np.where(~is_ring_list)[0]
    ring_indices = np.where(is_ring_list)[0]

    clusters = []
    clusters += cluster_peaks(peak_indices, ring_indices, centers, boxes, is_ring_list, r, extend)
    clusters += cluster_rings(ring_indices, centers, boxes, r, extend)
    return clusters
