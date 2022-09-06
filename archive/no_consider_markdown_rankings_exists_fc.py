"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

def create_idx_mapping(input_ids, cell_metadata, disable_print=False):
    """Create an in-between index map.
    
    Mapping will be used to index the features and labels.
    Aims to compare cells like so [code1, markdown1, code2]
    Adds in-between context for the markdown cell.

    Returns:
        index_map: list of 3-D tensor arrays
    """
    
    idx_map = []

    doc_start = 0
    
    for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Creating Index Mapping', disable=disable_print)):
        code_idxs = [idx
                    for idx, cell_id in enumerate(metadata)
                    if metadata[cell_id] == 'code']
        markdown_idxs = [idx
                        for idx, cell_id in enumerate(metadata)
                        if metadata[cell_id] == 'markdown']
        doc_map = [[doc_start+markdown_idx,
                    doc_start+code_idx,
                    doc_start+code_idx+1,
                    doc_start+code_idx+2,
                    doc_start+code_idx+3, 
                    doc_count]
                  for code_idx in code_idxs[:-3]
                  for markdown_idx in markdown_idxs]
        doc_start += len(metadata)
        idx_map.append(np.asarray(doc_map, dtype=np.int32))
    idx_map = np.concatenate(idx_map)
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
  print(all_excess_markdowns)
  
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
  markdown_uniques = []
  markdown_dupes = {}
  excess_markdowns = []
  
  abs_markdown_idxs = np.asarray(markdown_idxs) + doc_start
  markdowns = list(np.take(input_ids, abs_markdown_idxs))
  for idx, markdown in enumerate(markdowns): 
    if tuple(markdown) in markdown_uniques:  # Tuples support hashability
      orig_loc = markdown_uniques.index(tuple(markdown))  # Markdown idx of unique
      abs_orig_loc = abs_markdown_idxs[orig_loc]  # Change idx to absolute for idx map
      abs_dupe_loc = abs_markdown_idxs[idx]
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
    ordered_cells = df_orders.loc[doc_id]
    doc_order = np.asarray([ordered_cells.index(cell_id) for cell_id in cell_metadata[counter]])
    orders.append(doc_order)

  orders = np.concatenate(orders, axis=0)
  return orders

def remove_markdown_dupes_from_orders(orders, markdown_dupes):
  markdown_dupes_idx = [dupe_idx
                        for duped_markdowns in markdown_dupes.values()
                        for dupe_idx in duped_markdowns]
  return np.delete(orders, markdown_dupes_idx, axis=0)

def create_labels(idx_map, orders):
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

    labels = []
    markdown_ranks = np.take(orders, idx_map[:, 0])
    uo_code_ranks = np.take(orders, idx_map[:, 1])
    lo_code_ranks = np.take(orders, idx_map[:, 4])

    # Check if it exists 2 ranks above or 2 ranks below
    exists_above = np.where((markdown_ranks < uo_code_ranks - 1), 1, 0)
    exists_below = np.where((markdown_ranks > lo_code_ranks + 1), 1, 0)

    # Create existence labels as long as it exists in the given code block
    exists_label = np.where(((exists_above + exists_below) == 0), 1, 0)
    exists_label = exists_label.astype(np.int32)
    exists_label = np.expand_dims(exists_label, axis=-1)
    return exists_label

def add_markdown_dupe_labels(labels, idx_map, markdown_dupes):
  pass

def shuffle_and_adjust_class_imbalance(idx_map,
                                       labels,
                                       dataset_size,
                                       random_state):
  """Positive label is the miniority indices"""
  rng = np.random.default_rng(random_state)

  pos_indices = np.nonzero(labels == 1)[0]
  pos_mapping = np.take(idx_map, pos_indices, axis=0)

  idx_map = np.delete(idx_map, pos_mapping, axis=0)
  smallest_class_size = len(pos_mapping)
                     
  rng.shuffle(idx_map)
  idx_map = idx_map[:smallest_class_size]
  idx_map = np.concatenate([idx_map, pos_mapping], axis=0)

  rng.shuffle(idx_map)
  idx_map = idx_map[:dataset_size]
  return idx_map

def index_input_ids_to_map(input_ids, idx_map):
    markdown_input_ids = np.take(input_ids, indices=idx_map[:, 0], axis=0)
    uo_code_input_ids = np.take(input_ids, indices=idx_map[:, 1], axis=0)
    ui_code_input_ids = np.take(input_ids, indices=idx_map[:, 2], axis=0)
    li_code_input_ids = np.take(input_ids, indices=idx_map[:, 3], axis=0)
    lo_code_input_ids = np.take(input_ids, indices=idx_map[:, 4], axis=0)

    input_ids = (markdown_input_ids,
                 uo_code_input_ids,
                 ui_code_input_ids,
                 li_code_input_ids,
                 lo_code_input_ids)
    return input_ids

def add_special_tokens_and_masks(input_ids,
                                 tokenizer, 
                                 req_input_length,
                                 disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []

  for i in tqdm(range(len(input_ids[0])), disable=disable_print, desc='Adding Special Tokens'):
    input_id = ([tokenizer.cls_token_id]              
                + list(input_ids[0][i])
                + [tokenizer.sep_token_id]
                + list(input_ids[1][i])
                + [tokenizer('<c0>')['input_ids'][1]]
                + list(input_ids[2][i])
                + [tokenizer('<c1>')['input_ids'][1]]
                + list(input_ids[3][i])
                + [tokenizer('<c2>')['input_ids'][1]]
                + list(input_ids[4][i])
                + [tokenizer.sep_token_id])
    attention = [1] * len(input_id)
    no_attention = [0] * (req_input_length - len(attention))
    attention_mask = attention + no_attention
    input_id_padding = [tokenizer.pad_token_id] * (req_input_length - len(input_id))
    input_id = input_id + input_id_padding

    feature_input_ids.append(input_id)
    feature_attention_masks.append(attention_mask)   

  feature_input_ids = np.asarray(feature_input_ids, dtype=np.int32)
  feature_attention_masks = np.asarray(feature_attention_masks, dtype=np.int32)                     
  return feature_input_ids, feature_attention_masks