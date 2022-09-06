"""Creates features and labels to determine possible markdown location."""

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

def encode_text(text, tokenizer, verbose=0):
    '''Encodes text for the transformer model using the specified tokenizer.'''
    encoded_text = tokenizer(text,
                             max_length=30,
                             padding=False,
                             truncation=True,
                             add_special_tokens=False,
                             return_attention_mask=False)
    input_ids = encoded_text['input_ids']
    input_ids = np.asarray(input_ids, dtype='object')
    if verbose == 1:
      print('Text has been encoded.')
    return input_ids

def create_idx_map(cell_metadata):
    '''Create an in-between index map.
    
    Mapping will be used to index the features and labels.
    Aims to compare cells like so [code1, markdown1, code2]
    Adds in-between context for the markdown cell.

    Returns:
        index_map: list of 3-D tensor arrays
    '''
    
    idx_map = []
    doc_start = 0
    
    for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Creating Index Mapping')):
        code_idxs = [idx
                    for idx, cell_id in enumerate(metadata)
                    if metadata[cell_id] == 'code']
        markdown_idxs = [idx
                        for idx, cell_id in enumerate(metadata)
                        if metadata[cell_id] == 'markdown']   
        doc_map = [[doc_start+code_idx,
                    doc_start+code_idx+1,
                    doc_start+markdown_idx,
                    doc_start+code_idx+2,
                    doc_start+code_idx+3, 
                    doc_count]
                  for code_idx in code_idxs[:-3]
                  for markdown_idx in markdown_idxs]
        doc_start += len(metadata)
        idx_map.append(np.asarray(doc_map, dtype=np.int32))

    idx_map = np.concatenate(idx_map)
    return idx_map

def create_labels(doc_ids, cell_metadata, df_orders, idx_map):
    '''Creates labels by comparing the actual ordering with the index map.
    
    Args:
        doc_ids:
        cell_metadata:
        df_orders:
        idx_map:

    Returns:
        labels: 4-D binary array. A hot-end encoded output that determines if 
                the markdown cell exists above, below, or inbetween the two code cells    
    '''

    orders = []
    labels = []
    
    for counter, doc_id in enumerate(doc_ids):
        ordered_cells = df_orders.loc[doc_id]
        doc_order = np.asarray([ordered_cells.index(cell_id) for cell_id in cell_metadata[counter]])
        orders.append(doc_order)

    orders = np.concatenate(orders, axis=0)
    uo_code_ranks = np.take(orders, idx_map[:, 0])
    ui_code_ranks = np.take(orders, idx_map[:, 1])
    markdown_ranks = np.take(orders, idx_map[:, 2])
    li_code_ranks = np.take(orders, idx_map[:, 3])
    lo_code_ranks = np.take(orders, idx_map[:, 4])

    # Create outer labels
    upper_outer_label = np.where((markdown_ranks < uo_code_ranks), 1, 0)
    lower_outer_label = np.where((markdown_ranks > lo_code_ranks), 1, 0)

    # Use intermediate information to check where label exists
    exists_upper = np.where((markdown_ranks < ui_code_ranks), markdown_ranks, -1000)
    exists_lower = np.where((markdown_ranks > li_code_ranks), markdown_ranks, 1000)

    upper_inner_label = np.where((exists_upper > uo_code_ranks), 1, 0)
    lower_inner_label = np.where((exists_lower < lo_code_ranks), 1, 0)
    
    # Create center labels
    center_label = np.where(((upper_outer_label
                              + upper_inner_label
                              + lower_inner_label
                              + lower_outer_label)
                            == 0), 1, 0)

    labels = np.asarray([upper_outer_label,
                         upper_inner_label,
                         center_label,
                         lower_inner_label,
                         lower_outer_label], dtype=np.int32)
    labels = np.swapaxes(labels, 0, 1)
    return labels

def shuffle_and_adjust_class_imbalance(idx_map,
                                       labels,
                                       dataset_size,
                                       random_state):
  """All five clasess should have same probability of occuring
  
  Upper inner labels, center labels, and lower inner labels all
  share the same minority probabilitiy. Adjust the dataset 
  approrioately for this class inbalance.
  """

  rng = np.random.default_rng(random_state)
  ui_indices = np.nonzero(labels[:, 1] == 1)[0]
  ui_mapping = np.take(idx_map, ui_indices, axis=0)

  center_indices = np.nonzero(labels[:, 2] == 1)[0]
  center_mapping = np.take(idx_map, center_indices, axis=0)

  li_indices = np.nonzero(labels[:, 3] == 1)[0]
  li_mapping = np.take(idx_map, li_indices, axis=0)

  minority_indices = np.concatenate([ui_indices,
                                     center_indices,
                                     li_indices])
  minority_mapping = np.concatenate([ui_mapping,
                                     center_mapping,
                                     li_mapping])
  idx_map = np.delete(idx_map, minority_indices, axis=0)

  smallest_class_size = min([len(ui_mapping),
                             len(center_mapping),
                             len(li_mapping)])
                     
  rng.shuffle(idx_map)
  idx_map = idx_map[:smallest_class_size*2]
  idx_map = np.concatenate([idx_map, minority_mapping], axis=0)

  rng.shuffle(idx_map)
  idx_map = idx_map[:dataset_size]
  return idx_map

def index_input_ids_to_map(input_ids, idx_map):
    uo_code_input_ids = np.take(input_ids, indices=idx_map[:, 0], axis=0)
    ui_code_input_ids = np.take(input_ids, indices=idx_map[:, 1], axis=0)
    markdown_input_ids = np.take(input_ids, indices=idx_map[:, 2], axis=0)
    li_code_input_ids = np.take(input_ids, indices=idx_map[:, 3], axis=0)
    lo_code_input_ids = np.take(input_ids, indices=idx_map[:, 4], axis=0)

    input_ids = (uo_code_input_ids,
                 ui_code_input_ids,
                 markdown_input_ids,
                 li_code_input_ids,
                 lo_code_input_ids)
    return input_ids

def add_special_tokens_and_masks(input_ids,
                                 tokenizer, 
                                 req_input_length,
                                 verbose=1):
  feature_input_ids = []
  feature_attention_masks = []
  if verbose == 1:
    disable_tqdm = False
  else:
    disable_tqdm = True

  for i in tqdm(range(len(input_ids[0])), disable=disable_tqdm, desc='Adding Special Tokens'):
    input_id = ([tokenizer.cls_token_id]              
                + list(input_ids[2][i])
                + [tokenizer.sep_token_id]
                + list(input_ids[0][i])
                + [tokenizer('<c0>')['input_ids'][1]]
                + list(input_ids[1][i])
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

  # np.concatenate([], axis=1, dtype=np.int32)