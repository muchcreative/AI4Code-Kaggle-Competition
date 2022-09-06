"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

def adjust_input_ids_for_ordering(input_ids, cell_metadata):
  doc_start = 0
  for metadata in cell_metadata:
    doc_end = len(metadata)
    doc_code_md_locs = list(metadata.values())
    code_cell_count = doc_code_md_locs.count('code')
    code_len, md_len = get_optimal_order_code_md_lens(code_cell_count)
    input_ids[doc_start:doc_start+doc_end] = concat_long_input_ids(input_ids[doc_start:doc_start+doc_end],
                                                                   doc_code_md_locs,
                                                                   code_len,
                                                                   md_len)
    doc_start += doc_end
  return input_ids

def get_optimal_order_code_md_lens(code_cell_count):
  # Must be even numbers for assertion
  if code_cell_count <= 6:  # 6 or less code cells per block; 489 seq_len
    code_len = 60
    md_len = 120
  else:  # 7-8 code cells per block; 512 seq_len
    code_len = 50
    md_len = 100
  return code_len, md_len
    
def concat_long_input_ids(input_ids,
                          code_markdown_locs,
                          code_len,
                          md_len):
  """Converts long input ids to concatenated start and end parts."""

  one_side_md_len = int(md_len/2)
  one_side_code_len = int(code_len/2)

  for idx, line in enumerate(input_ids):
    if len(line) > code_len and code_markdown_locs[idx] == 'code':
      input_ids[idx] = line[:one_side_code_len] + line[-one_side_code_len:]
      assert len(input_ids[idx]) == code_len, 'You summed the input ids, please double check concatenation.'
    elif len(line) > md_len and code_markdown_locs[idx] == 'markdown':
      input_ids[idx] = line[:one_side_md_len] + line[-one_side_md_len:]
      assert len(input_ids[idx]) == md_len, 'You summed the markdown input ids, please double check concatenation.'
  return input_ids

def create_idx_map(cell_metadata, disable_print=False): 
    idx_map = []
    md_locs = []
    doc_start = 0
    
    for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Creating Index Mapping', disable=disable_print)):
      doc_map = []
      doc_code_md_locs = list(metadata.values())

      code_idxs = [idx + doc_start
                   for idx, cell_id in enumerate(metadata)
                   if metadata[cell_id] == 'code']
      md_idxs = [idx + doc_start
                 for idx, cell_id in enumerate(metadata)
                 if metadata[cell_id] == 'markdown']

      code_idxs = np.asarray(code_idxs, dtype=np.int32)
      code_cell_count = len(code_idxs)
      md_idxs = np.asarray(md_idxs, dtype=np.int32)
      md_cell_count = len(md_idxs)

      if code_cell_count >= 9:  # Starts and Ends Collect All, Mid Collect Spreads
        code_cells_per_block = 8
        start_block = code_idxs[:code_cells_per_block]
        end_block = code_idxs[-code_cells_per_block:]
        for md_idx in md_idxs:
          for start_pos in range(4):
            doc_map.append(np.insert(start_block, start_pos, md_idx, axis=0))
          for mid_pos in range(code_cell_count-code_cells_per_block):
            doc_map.append(np.insert(code_idxs[mid_pos:mid_pos+code_cells_per_block], 4, md_idx, axis=0))
          for end_pos in range(4, 9):
            doc_map.append(np.insert(end_block, end_pos, md_idx, axis=0))
        md_loc = (list(range(4)) + [4]*(code_cell_count-7) + list(range(5, code_cells_per_block+1))) * md_cell_count
        md_locs.append(md_loc)

      else:  # If less than or equal to 8 code cells, can put block in one input
        doc_map = [np.insert(code_idxs, i, md_idx, axis=0)
                  for md_idx in md_idxs
                  for i in range(len(code_idxs)+1)]
        md_loc = list(range(code_cell_count+1)) * md_cell_count
        md_locs.append(md_loc)
      
      idx_map.append(np.asarray(doc_map, dtype=np.int32))
      doc_start += len(metadata)
    md_locs = np.concatenate(md_locs, dtype=np.int32)
    return idx_map, md_locs

def get_orders(doc_ids, cell_metadata, df_orders):
  orders = []

  for counter, doc_id in enumerate(doc_ids):
    ordered_cell_ids = df_orders.loc[doc_id]
    ordered_cell_types = [cell_metadata[counter][cell_id] for cell_id in ordered_cell_ids]
    doc_cell_ids = list(cell_metadata[counter].keys())

    ranks = {}
    current_rank = 0  # First position is 1 or 0.5
    in_markdown_block = False

    for idx, cell_id in enumerate(ordered_cell_ids):
      if ordered_cell_types[idx] == 'code':
        if current_rank % 1 == 0:
          current_rank += 1
        else:
          current_rank += 0.5
        ranks[cell_id] = current_rank
        in_markdown_block = False

      elif ordered_cell_types[idx] == 'markdown' and not in_markdown_block:
        current_rank += 0.5
        ranks[cell_id] = current_rank
        in_markdown_block = True
        
      elif ordered_cell_types[idx] == 'markdown' and in_markdown_block:
        ranks[cell_id] = current_rank
        markdown_block = True

    unordered_ranks = [ranks[cell_id] for cell_id in doc_cell_ids]
    orders.append(unordered_ranks)

  orders = np.concatenate(orders, axis=0)
  return orders

def create_labels(idx_map, orders):
  labels = []
  for doc_map in idx_map:
    doc_orders = np.take(orders, doc_map, axis=0)
    arr_width = doc_orders.shape[1]
    is_ordered_mask = (doc_orders == np.sort(doc_orders))
    sum_mask = np.sum(is_ordered_mask, axis=1)
    doc_labels = np.where(sum_mask == arr_width, 1, 0)
    labels.append(doc_labels)
  labels = np.concatenate(labels, dtype=np.float32)
  return labels

def shuffle_and_adjust_class_imbalance(idx_map,
                                       md_locs, 
                                       labels,
                                       samples_per_file,
                                       random_state):
  """Positive label is the miniority indices"""
  rng = np.random.default_rng(random_state)

  flattened_idx_map = [block
                       for doc_map in idx_map
                       for block in doc_map]
  flattened_idx_map = np.asarray(flattened_idx_map, dtype=object)

  pos_indices = np.nonzero(labels == 1)[0]
  pos_idx_map = np.take(flattened_idx_map, pos_indices, axis=0)
  pos_md_locs = np.take(md_locs, pos_indices, axis=0)
  pos_labels = np.take(labels, pos_indices, axis=0)

  neg_idx_map = np.delete(flattened_idx_map, pos_indices, axis=0)
  neg_md_locs = np.delete(md_locs, pos_indices, axis=0)
  neg_labels = np.delete(labels, pos_indices, axis=0)
  smallest_class_size = len(pos_labels)
  
  # Reduce negative class size
  dummy_range = np.arange(0, len(neg_labels))
  rng.shuffle(dummy_range)
  neg_idx_map = np.take(neg_idx_map, dummy_range, axis=0)[:smallest_class_size]
  neg_md_locs = np.take(neg_md_locs, dummy_range, axis=0)[:smallest_class_size]
  neg_labels = np.take(neg_labels, dummy_range, axis=0)[:smallest_class_size]
  
  # Add back positive labels
  idx_map = np.concatenate([pos_idx_map, neg_idx_map], axis=0)
  md_locs = np.concatenate([pos_md_locs, neg_md_locs], axis=0)
  labels = np.concatenate([pos_labels, neg_labels], axis=0)

  # Take a finally shuffle and truncate to the specified dataset size
  dummy_range = np.arange(0, len(idx_map))
  rng.shuffle(dummy_range)
  idx_map = np.take(idx_map, dummy_range, axis=0)[:samples_per_file]
  md_locs = np.take(md_locs, dummy_range, axis=0)[:samples_per_file]
  labels = np.take(labels, dummy_range, axis=0)[:samples_per_file]
  labels = np.reshape(labels, (len(labels), 1))
  return idx_map, md_locs, labels

def index_input_ids_to_map(input_ids, idx_map):
  mapped_input_ids = [np.take(input_ids, block, axis=0)
                      for block in idx_map]
  return mapped_input_ids

def add_special_tokens_and_masks(input_ids,
                                 md_locs, 
                                 tokenizer, 
                                 seq_len=512,
                                 disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []
  code_split_token = [tokenizer('<c>')['input_ids'][1]]
  md_split_token = [tokenizer('<m>')['input_ids'][1]]

  for count, block in enumerate(tqdm(input_ids, disable=disable_print, desc='Adding Special Tokens')):
    encoded_block = []     
    for idx, input_id in enumerate(block):
      if idx == md_locs[count]:
        encoded_block.extend(md_split_token + input_id)
      else:
        encoded_block.extend(code_split_token + input_id)
     
    encoded_input_id = [tokenizer.cls_token_id] + encoded_block + [tokenizer.sep_token_id]
    attention = [1] * len(encoded_input_id)
    no_attention = [0] * (seq_len - len(attention))
    attention_mask = attention + no_attention
    input_id_padding = [tokenizer.pad_token_id] * (seq_len - len(encoded_input_id))
    encoded_input_id = encoded_input_id + input_id_padding

    feature_input_ids.append(encoded_input_id)
    feature_attention_masks.append(attention_mask)   

  feature_input_ids = np.asarray(feature_input_ids, dtype=np.int32)
  feature_attention_masks = np.asarray(feature_attention_masks, dtype=np.int32)                     
  return feature_input_ids, feature_attention_masks