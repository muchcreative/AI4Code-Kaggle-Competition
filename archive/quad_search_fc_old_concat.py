"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import copy
import random

import itertools
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def get_quadrant_locs(doc_ids, cell_metadata, df_orders):
  md_quadrant_locs = {}

  for count, doc_id in enumerate(doc_ids):
    metadata = cell_metadata[count]
    doc_orders = df_orders[doc_id]
    code_cell_count = list(metadata.values()).count('code')
    quadrant_splits = create_quadrant_splits(code_cell_count)

    code_location = 0
    for cell_id in doc_orders:
      if metadata[cell_id] == 'markdown':
        for idx, quadrant_split in enumerate(quadrant_splits[1:]):
          if quadrant_split > code_location or code_cell_count == quadrant_split:
            md_quadrant_locs[cell_id] = idx
            break
      else:
        code_location += 1
  return md_quadrant_locs

def create_quadrant_splits(code_cell_count):
  quarter_loc, remainder = divmod(code_cell_count, 4)
  quadrant_splits = np.arange(0, code_cell_count-remainder+0.1, step=quarter_loc, dtype=np.int32)
  for i in range(0, remainder):
    quadrant_splits[i+1:] = quadrant_splits[i+1:] + 1
  return quadrant_splits
    
def collect_md_code_groupings(md_quadrant_locs, cell_metadata):
  rng = np.random.default_rng()
  md_code_groupings = []
  labels = []
  doc_start = 0

  for metadata in tqdm(cell_metadata, desc='Collecting Markdown Code Groupings'):
    doc_length = len(metadata)
    markdown_ids = {}
    code_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        markdown_ids[cell_id] = idx+doc_start

    quarter_loc, remainder = divmod(code_count, 4)
    quadrant_splits = create_quadrant_splits(code_count)

    for cell_id, md_abs_idx in markdown_ids.items():
      code_vectors = create_doc_code_representation(quarter_loc,
                                                    quadrant_splits,
                                                    rng,
                                                    doc_start)
      for code_vector in code_vectors:
        md_code_groupings.append([md_abs_idx, code_vector])
        labels.append(md_quadrant_locs[cell_id])
    doc_start += doc_length
  labels = np.asarray(labels, dtype=np.float32)
  return md_code_groupings, labels

def create_doc_code_representation(quarter_loc,
                                   quadrant_splits,
                                   rng,
                                   doc_start=0):
  """Code count must always be equal to or greater than 4
  doc_start is for creating features all at once instead of per document id
  """
  code_rep = []

  if quarter_loc <= 4:
    for i in range(4):
      quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1])
      code_rep.append(quadrant_vector + doc_start)
    code_rep = [code_rep]  # Check this

  else:
    if quarter_loc == 5:
      max_samples = 1
    elif quarter_loc == 6:
      max_samples = 5
    else:  # 7 or more, around 21 combinations avaliable
      max_samples = 7

    quadrant_vectors = []
    for i in range(4):
      if quarter_loc >= 15:  # This will be handled in inference
        sampled_quadrant = np.sort(rng.choice(np.arange(quadrant_splits[i], quadrant_splits[i+1]), size=15, replace=False))
        combination_vectors = np.asarray(list(itertools.combinations(sampled_quadrant, 5)))
      else:
        combination_vectors = np.asarray(list(itertools.combinations(np.arange(quadrant_splits[i], quadrant_splits[i+1]), 5)))
      quadrant_vectors.append(rng.choice(combination_vectors, max_samples, replace=False))
    code_rep = list(np.stack(quadrant_vectors, axis=1) + doc_start)
  return code_rep

def adjust_input_ids_for_quad_len(input_ids, cell_metadata):
  doc_start = 0
  for metadata in cell_metadata:
    doc_end = len(metadata)
    doc_code_md_locs = list(metadata.values())
    code_cell_count = doc_code_md_locs.count('code')
    code_len, md_len = get_optimal_quad_code_md_lens(code_cell_count)
    input_ids[doc_start:doc_start+doc_end] = concat_long_input_ids(input_ids[doc_start:doc_start+doc_end],
                                                                   doc_code_md_locs,
                                                                   code_len,
                                                                   md_len)
    doc_start += doc_end
  return input_ids

def concat_long_input_ids(input_ids,
                          code_markdown_locs,
                          max_code_length,
                          max_markdown_length):
  """Converts long input ids to concatenated start and end parts."""

  one_side_markdown_length = int(max_markdown_length/2)
  one_side_code_length = int(max_code_length/2)

  for idx, line in enumerate(input_ids):
    if len(line) > max_code_length and code_markdown_locs[idx] == 'code':
      input_ids[idx] = line[:one_side_code_length] + line[-one_side_code_length:]
    elif len(line) > max_code_length and code_markdown_locs[idx] == 'markdown':
      input_ids[idx] = line[:one_side_markdown_length] + line[-one_side_markdown_length:]
  return input_ids

def get_optimal_quad_code_md_lens(code_cell_count):
  if code_cell_count <= 4: # 1 code cell per quadrant
    code_len = 60
    md_len = 120
  elif code_cell_count <= 8:  # 2 code cells per quadrant
    code_len = 49
    md_len = 114
  elif code_cell_count <= 12:  # 3 code cells per quadrant
    code_len = 33
    md_len = 110
  elif code_cell_count <= 16:  # 4 code cells per quadrant
    code_len = 26
    md_len = 90
  else:  # 5 code cells per quadrant
    code_len = 22
    md_len = 66
  return code_len, md_len

def create_quadrant_features(input_ids,
                             pointwise_groupings,
                             tokenizer,
                             seq_len=512,
                             disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []
  code_split_token = tokenizer('<c>')['input_ids'][1]
  
  for md_idx, code_grouping in tqdm(pointwise_groupings, desc='Assembling Input Ids and Attention Masks', disable=disable_print):
    code_input_id = []
    markdown_input_id = ([tokenizer.sep_token_id]
                          + list(input_ids[md_idx])
                          + [tokenizer.cls_token_id])
                          
    if len(code_grouping) == 4:  # 4 quadrants avaliable
      for code_quadrant in code_grouping:
        quadrant = np.take(input_ids, code_quadrant)
        quadrant = np.concatenate(quadrant, axis=0)
        code_input_id +=  list(quadrant) + [code_split_token]
    else:  # Less than 4 code cells
      assert len(code_grouping) <= 3, f'Code grouping contains less than 4 quadrants. Currently at {len(code_grouping)}. Check filtering.'

    input_id = markdown_input_id + code_input_id[:-1] + [tokenizer.sep_token_id]
    attention = [1] * len(input_id)
    no_attention = [0] * (seq_len - len(attention))
    attention_mask = attention + no_attention
    
    input_id_padding = [tokenizer.pad_token_id] * (seq_len - len(input_id))
    input_id = input_id + input_id_padding
    feature_input_ids.append(input_id)
    feature_attention_masks.append(attention_mask)   

  feature_input_ids = np.asarray(feature_input_ids, dtype=np.int32)
  feature_attention_masks = np.asarray(feature_attention_masks, dtype=np.int32)      
  return [feature_input_ids, feature_attention_masks]

def hot_end_encode_labels(labels):
  labels = tf.one_hot(labels, 4, axis=1, dtype=np.float32).numpy()
  return labels

def sample_balance_shuffle(features, labels, samples_per_file):
  rng = np.random.default_rng()
  
  largest_class_idx = np.where(labels[:, 0] == 1)[0]
  largest_class_input_ids = np.take(features[0], largest_class_idx, axis=0)
  largest_class_attention_masks = np.take(features[1], largest_class_idx, axis=0)
  
  minority_class_size = np.min([sum(labels[:, 1]),
                                sum(labels[:, 2]),
                                sum(labels[:, 3])]).astype(np.int32)
  
  shuffler = np.arange(len(largest_class_idx))
  rng.shuffle(shuffler)
  largest_class_input_ids = np.take(largest_class_input_ids, shuffler, axis=0)[:minority_class_size]
  largest_class_attention_masks = np.take(largest_class_attention_masks, shuffler, axis=0)[:minority_class_size]
  largest_class_labels = np.take(labels, largest_class_idx, axis=0)[:minority_class_size]
  
  features[0] = np.delete(features[0], largest_class_idx, axis=0)
  features[1] = np.delete(features[1], largest_class_idx, axis=0)
  labels = np.delete(labels, largest_class_idx, axis=0)

  features[0] = np.concatenate([features[0], largest_class_input_ids])
  features[1] = np.concatenate([features[1], largest_class_attention_masks])
  labels = np.concatenate([labels, largest_class_labels], axis=0)

  shuffler = np.arange(len(labels))
  rng.shuffle(shuffler)
  features[0] = np.take(features[0], shuffler, axis=0)[:samples_per_file]
  features[1] = np.take(features[1], shuffler, axis=0)[:samples_per_file]
  labels = np.take(labels, shuffler, axis=0)[:samples_per_file]
  return tuple(features), labels