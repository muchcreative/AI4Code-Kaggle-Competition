"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def create_labels(idx_map, orders, code_cell_count):
    """Creates labels by comparing the actual ordering with the index map.
    
    Args:
        doc_ids:
        cell_metadata:
        df_orders:
        idx_map:

    Returns:
        labels: 4-D binary array. A hot-end encoded output that determines if 
                the markdown cell exists above, below, or inbetween the two code cells    
    """
  
    markdown_ranks = np.take(orders, idx_map[:, 0])
    code_ranks = [np.take(orders, idx_map[:, i])
                  for i in range(1, code_cell_count+1)]
    labels = [np.where((markdown_ranks == code_ranks[i]-1), 1, 0)
              for i in range(code_cell_count)]
    labels.append(np.where((markdown_ranks == code_ranks[-1]+1), 1, 0))    
    labels = np.stack(labels, axis=1)
    return labels

def adjust_labels_for_dupes(labels,
                            markdown_dupes,
                            idx_map,
                            orders,
                            code_cell_count):
  for original, dupes in markdown_dupes.items():
    abs_idx = np.where(idx_map[:,0] == original)[0]
    sample_map = idx_map[abs_idx, :]
    for dupe in dupes:
      sample_map[:, 0] = dupe
      sample_labels = create_labels(sample_map, orders, code_cell_count)
      labels[abs_idx] += sample_labels
    
  labels = np.where(labels >= 1, 1, 0)
  labels = labels.astype(np.float32)
  return labels

def shuffle_and_remove_zero_labels(idx_map,
                                   labels,
                                   samples_per_file,
                                   random_state):
  """Delete rows with no labels in them then shuffle.
  
  Already know the markdown exists in the given block, need to predict
  location of it"""
  zero_indices = np.where(~labels.any(axis=1))[0]  # Delete rows with empty labels
  idx_map = np.delete(idx_map, zero_indices, axis=0)
  labels = np.delete(labels, zero_indices, axis=0)

  rng = np.random.default_rng(random_state)
  dummy_range = np.arange(0, len(labels))
  rng.shuffle(dummy_range)

  idx_map = np.take(idx_map, dummy_range, axis=0)[:samples_per_file]
  labels = np.take(labels, dummy_range, axis=0)[:samples_per_file]
  return idx_map, labels