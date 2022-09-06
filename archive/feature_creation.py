"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def create_idx_map(cell_metadata, code_cell_count, disable_print=False):
    """Create an in-between index map.
    
    Mapping will be used to index the features and labels.
    Aims to compare cells like so [code1, markdown1, code2]
    Adds in-between context for the markdown cell.

    Start and end locations are for the remaining of the code cells.
    Helps determine how deep it is and how much left it has.

    If for categorical addition
    #   + [np.digitize((code_idx)/(len(code_idxs))*100, loc_bins)]  # Start location of the code cells
    #   + [np.digitize((code_idx+code_cell_count-1)/(len(code_idxs))*100, loc_bins)]  # End location of the code cells
    #   + [doc_count]

    Returns:
        index_map: list of 3-D tensor arrays
    """
    
    idx_map = []

    doc_start = 0
    loc_bins = [25, 50, 75]
    
    for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Creating Index Mapping', disable=disable_print)):
      code_idxs = [idx
                   for idx, cell_id in enumerate(metadata)
                   if metadata[cell_id] == 'code']
      markdown_idxs = [idx
                       for idx, cell_id in enumerate(metadata)
                       if metadata[cell_id] == 'markdown']
      doc_map = [[doc_start+markdown_idx]
                  + [doc_start+code_idx+i for i in range(code_cell_count)]
                  for code_idx in code_idxs[:-(code_cell_count-1)]
                  for markdown_idx in markdown_idxs]
      doc_start += len(metadata)
      idx_map.append(np.asarray(doc_map, dtype=np.int32))
    idx_map = np.concatenate(idx_map, axis=0)
    return idx_map

def remove_markdown_dupes_from_idx_map(idx_map, input_ids, cell_metadata, disable_print=False):
  all_markdown_dupes = {} # Used to adjust for new rankings when you create labels
  all_excess_markdowns = set()  # Removable from idx_map
  dupe_idxs = []
  doc_start = 0

  for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Removing Markdown Dupes', disable=disable_print)):
    markdown_idxs = [idx
                     for idx, cell_id in enumerate(metadata)
                     if metadata[cell_id] == 'markdown']
    markdown_dupes, excess_markdowns = collect_markdown_duplicates(input_ids, markdown_idxs, doc_start)
    all_markdown_dupes.update(markdown_dupes)
    all_excess_markdowns.update(excess_markdowns)
    doc_start += len(metadata)

  for excess_markdown in all_excess_markdowns:
    dupe_idxs.append(np.where(idx_map[:,0] == excess_markdown)[0])

  dupe_idxs = np.concatenate(dupe_idxs)
  idx_map = np.delete(idx_map, dupe_idxs, axis=0)
  return idx_map, all_markdown_dupes

def collect_markdown_duplicates(input_ids, markdown_idxs, doc_start):
  """Collect markdowns that can be dropped
  and then their alternative mapping as
  {markdown_idx: [dupe1_idx, dupe2_idx]...}
  """
  markdown_dupes = {}
  markdown_uniques = []
  excess_markdowns = []
  
  abs_markdown_idxs = np.asarray(markdown_idxs) + doc_start
  markdowns = list(np.take(input_ids, abs_markdown_idxs))

  for count, markdown in enumerate(markdowns): 
    if tuple(markdown) in markdown_uniques:  # Tuples support hashability
      orig_loc = markdown_uniques.index(tuple(markdown))
      
      abs_orig_loc = abs_markdown_idxs[orig_loc]
      abs_dupe_loc = abs_markdown_idxs[count]
      while input_ids[abs_dupe_loc] != input_ids[abs_orig_loc]:  # If multiple duplicates, must shift appropriately, lazy execution
        abs_orig_loc += 1

      excess_markdowns.append(abs_dupe_loc)
      if abs_orig_loc in markdown_dupes:
         markdown_dupes[abs_orig_loc].append(abs_dupe_loc)
      else:
         markdown_dupes[abs_orig_loc] = [abs_dupe_loc]

    else:
      markdown_uniques.append(tuple(markdown))
  return markdown_dupes, excess_markdowns

def get_orders(doc_ids, cell_metadata, df_orders):
  orders = []

  for counter, doc_id in enumerate(doc_ids):
    ordered_cell_ids = df_orders.loc[doc_id]
    ordered_cell_types = [cell_metadata[counter][cell_id] for cell_id in ordered_cell_ids]
    doc_cell_ids = list(cell_metadata[counter].keys())

    ranks = {}
    current_rank = -1  # Must be negative so markdowns in the same block share rank
    in_markdown_block = False

    for idx, cell_id in enumerate(ordered_cell_ids):
      if ordered_cell_types[idx] == 'code':
        current_rank += 1
        ranks[cell_id] = current_rank
        in_markdown_block = False
        
      elif ordered_cell_types[idx] == 'markdown' and not in_markdown_block:
        current_rank += 1
        ranks[cell_id] = current_rank
        in_markdown_block = True
        
      elif ordered_cell_types[idx] == 'markdown' and in_markdown_block:
        ranks[cell_id] = current_rank
        markdown_block = True

    unordered_ranks = [ranks[cell_id] for cell_id in doc_cell_ids]
    orders.append(unordered_ranks)

  orders = np.concatenate(orders, axis=0)
  return orders

def remove_markdown_dupes_from_orders(orders, markdown_dupes):
  markdown_dupes_idx = [dupe_idx
                        for duped_markdowns in markdown_dupes.values()
                        for dupe_idx in duped_markdowns]
  return np.delete(orders, markdown_dupes_idx, axis=0)

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
    uo_code_ranks = np.take(orders, idx_map[:, 1])
    lo_code_ranks = np.take(orders, idx_map[:, code_cell_count])

    # Check if it exists 2 ranks above or 2 ranks below
    exists_above = np.where((markdown_ranks < uo_code_ranks - 1), 1, 0)
    exists_below = np.where((markdown_ranks > lo_code_ranks + 1), 1, 0)

    # Create existence labels as long as it exists in the given code block
    exists_label = np.where(((exists_above + exists_below) == 0), 1, 0)
    exists_label = np.expand_dims(exists_label, axis=-1)
    return exists_label

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

def shuffle_and_adjust_class_imbalance(idx_map,
                                       labels,
                                       dataset_size,
                                       random_state):
  """Positive label is the miniority indices"""
  rng = np.random.default_rng(random_state)

  pos_indices = np.nonzero(labels == 1)[0]
  pos_idx_map = np.take(idx_map, pos_indices, axis=0)
  pos_labels = np.take(labels, pos_indices, axis=0)

  neg_idx_map = np.delete(idx_map, pos_indices, axis=0)
  neg_labels = np.delete(labels, pos_indices, axis=0)
  smallest_class_size = len(pos_labels)
  
  # Reduce negative class size
  dummy_range = np.arange(0, len(neg_labels))
  rng.shuffle(dummy_range)
  neg_idx_map = np.take(neg_idx_map, dummy_range, axis=0)[:smallest_class_size]
  neg_labels = np.take(neg_labels, dummy_range, axis=0)[:smallest_class_size]
  
  # Add back positive labels
  idx_map = np.concatenate([pos_idx_map, neg_idx_map], axis=0)
  labels = np.concatenate([pos_labels, neg_labels], axis=0)

  # Take a finally shuffle and truncate to the specified dataset size
  dummy_range = np.arange(0, len(idx_map))
  rng.shuffle(dummy_range)
  idx_map = np.take(idx_map, dummy_range, axis=0)[:dataset_size]
  labels = np.take(labels, dummy_range, axis=0)[:dataset_size]
  return idx_map, labels

def index_input_ids_to_map(input_ids, idx_map, code_cells_per_block):
  input_ids = [np.take(input_ids, indices=idx_map[:, i], axis=0)
               for i in range(code_cells_per_block+1)]
  input_ids = tuple(input_ids)
  return input_ids

def convert_locs_to_list_tokens(idx_map, list_loc_tokens):
  start_token = [list_loc_tokens[bin_loc] for bin_loc in idx_map[:, -3]]
  end_token = [list_loc_tokens[bin_loc] for bin_loc in idx_map[:, -2]]
  return start_token, end_token

def convert_locs_to_hot_ends(idx_map):
  start_hot_end = tf.one_hot(idx_map[:, -3], 4, dtype=tf.float64)
  end_hot_end = tf.one_hot(idx_map[:, -2], 4, dtype=tf.float64)
  hot_end_loc_inputs = tf.concat((start_hot_end, end_hot_end), axis=-1)
  return hot_end_loc_inputs

def add_special_tokens_and_masks(input_ids,
                                 tokenizer, 
                                 sequence_length,
                                 code_cells_per_block,
                                 disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []

  for i in tqdm(range(len(input_ids[0])), disable=disable_print, desc='Adding Special Tokens'):
    markdown_section = ([tokenizer.cls_token_id]
                         + list(input_ids[0][i])
                         + [tokenizer.sep_token_id])
    code_section = []
    for code_loc in range(1, code_cells_per_block):
      code_section += list(input_ids[code_loc][i]) + [tokenizer('<c>')['input_ids'][1]]
                    
    last_code_section = list(input_ids[code_cells_per_block][i]) + [tokenizer.sep_token_id]
    input_id = markdown_section + code_section + last_code_section

    attention = [1] * len(input_id)
    no_attention = [0] * (sequence_length - len(attention))
    attention_mask = attention + no_attention
    input_id_padding = [tokenizer.pad_token_id] * (sequence_length - len(input_id))
    input_id = input_id + input_id_padding
    
    feature_input_ids.append(input_id)
    feature_attention_masks.append(attention_mask)   

  feature_input_ids = np.asarray(feature_input_ids, dtype=np.int32)
  feature_attention_masks = np.asarray(feature_attention_masks, dtype=np.int32)                     
  return feature_input_ids, feature_attention_masks

def create_idx_map_by_quadrant(md_quadrants,
                              code_markdown_locs,
                              code_cells_per_block, 
                              disable_print=False):
  idx_map = []
  md_records = []
  record_len = 0
  code_locs = np.where(code_markdown_locs == 'code')[0]
  md_locs = np.where(code_markdown_locs == 'markdown')[0]
  code_cell_count = len(code_locs)
  quarter_loc, remainder = divmod(code_cell_count, 4)
  
  quadrant_splits = np.arange(0, code_cell_count+0.1, quarter_loc, dtype=np.int32)
  for i in range(remainder):
    quadrant_splits[1+i:] = quadrant_splits[1+i:] + 1
    
  for md_idx, quad_pred in enumerate(tqdm(md_quadrants, desc='Creating Quadrant Mapping', disable=disable_print)):
    best_quadrants = quad_pred[0].astype(np.int32)
    for quadrant in best_quadrants:
      quadrant_start = quadrant_splits[quadrant]
      quadrant_end = quadrant_splits[quadrant+1]
      code_idxs = list(range(quadrant_start, quadrant_end))
      doc_map = [[md_locs[md_idx]]
                 + [code_idx+i for i in range(code_cells_per_block)]
                 for code_idx in code_idxs[:-(code_cells_per_block-1)]]
      
      record_len += len(doc_map)
      md_records.append(record_len)
      idx_map.append(np.asarray(doc_map, dtype=np.int32))
  
  md_records = np.asarray(md_records, dtype=np.int32)
  idx_map = np.concatenate(idx_map, axis=0)
  return idx_map, md_records

# idx_map, markdown_dupes = feature_creation.remove_markdown_dupes_from_idx_map(idx_map,
#                                                                               input_ids,
#                                                                               cell_metadata)
# labels = feature_creation.create_labels(idx_map, orders, code_cells_per_block)
# labels = feature_creation.adjust_labels_for_dupes(labels,
#                                                   markdown_dupes,
#                                                   idx_map,
#                                                   orders,
#                                                   code_cells_per_block)