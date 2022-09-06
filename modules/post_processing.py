'''Converts Data to TFRecords Format.'''

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from bisect import bisect

def create_idx_map_from_quadrant(md_idx, quadrant_range, code_cell_count):
  code_cell_idxs = np.arange(code_cell_count)
  idx_maps = []
  md_locs = []
  
  for md_pos in range(quadrant_range[0], quadrant_range[-1]+2):  # +2, np.inserts one back and need the last idx
    full_range = np.insert(code_cell_idxs, md_pos, md_idx)
    md_loc = np.nonzero(full_range==md_idx)[0][0]
    upper_section_len = len(full_range[md_loc+1:])
    lower_section_len = len(full_range[:md_loc])

    if upper_section_len >= 4 and lower_section_len >= 4:  # Markdown is in the center
      idx_maps.append(full_range[md_pos-4:md_pos+5])
      md_locs.append(4)
    elif upper_section_len > lower_section_len:  # In First Quadrant
      idx_maps.append(full_range[:9])  
      md_locs.append(md_pos)
    else:  # In Last Quadrant
      idx_maps.append(full_range[-9:]) 
      md_locs.append(8+md_pos-code_cell_count)
  idx_maps = np.vstack(idx_maps).astype(np.int32)
  return idx_maps, md_locs

def convert_range_to_doc_pos(quad_range):
  doc_pos = np.insert(quad_range, 0, quad_range[0]-1)
  return doc_pos

def assemble_doc_order(code_ids, markdown_ids, markdown_ranks):
  """Assembles the document order using the given markdown ranks.

  Loops through the ordered code ids. If a markdown has a rank
  in that location it will add the respective markdowns to that location.
  The order that the markdowns are added are done in the order that they appear.
  If no markdown exists in that location, only the respective code id will be added.

  Args:
    code_ids: pandas index of the ordered code ids
    markdown_ids: numpy array of the unordered markdown ids
    ranks: rank of each markdown id with each element location corresponding 
           to its markdown id location
  
  Returns:
    cell_order: list given as the combined ordered code and markdowns ids.
  """

  cell_order = []
  for i, code_id in enumerate(code_ids):
    markdown_loc = np.where(markdown_ranks == i)[0]
    cell_order.append(code_id)
    if markdown_loc.size != 0:
      cell_order.extend(np.take(markdown_ids, markdown_loc))
    
  # Check if there were markdowns before the first code cell with rank -1
  markdown_loc = np.where(markdown_ranks == -1)[0]
  if markdown_loc.size != 0:
    first_mds = np.take(markdown_ids, markdown_loc)
    cell_order = np.insert(cell_order, 0, first_mds, axis=0)
  return cell_order

def mid_insertion(code_ids, markdown_ids):
  """For un-implemented models, insert markdown ids in mid-section"""
  mid_section = int(round(len(code_ids+markdown_ids)/2))
  cell_order = code_ids[:mid_section] + markdown_ids + code_ids[mid_section:]
  return cell_order

def calculate_kendall_tau(ground_truth, predictions):
  """Calculates Kendall Tau between ground truth and predictions.

  Args:
    ground_truth: Pandas Series with doc ids as indexes and values as
                  a list of the ordered code and markdown ids.
    predictions: Pandas Series with doc ids as indexes and values as
                 a list of the ordered code and markdown ids.

  Returns:
    kendall_tau_score: float
  """
  
  total_inversions = 0  # total inversions in predicted ranks across all instances
  total_2max = 0  # maximum possible inversions across all instances
  for gt, pred in zip(ground_truth, predictions):
    ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
    total_inversions += count_inversions(ranks)
    n = len(gt)
    total_2max += n * (n - 1)
  return 1 - 4 * total_inversions / total_2max

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions

def get_ordered_df(doc_id, df, df_orders):
  """Creates the ordered pandas dataframe with the ordered text."""
  return df.loc[doc_id].loc[df_orders[doc_id]]