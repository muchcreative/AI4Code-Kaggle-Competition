"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import copy
import random

import itertools
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def get_md_pct_ranks(doc_ids, cell_metadata, df_orders):
  md_pct_ranks = {}

  for count, doc_id in enumerate(doc_ids):
    metadata = cell_metadata[count]
    doc_orders = df_orders[doc_id]
    doc_length = len(metadata)

    for idx, cell_id in enumerate(doc_orders):
      if metadata[cell_id] == 'markdown':
        md_pct_ranks[cell_id] = idx/doc_length
  return md_pct_ranks

def collect_md_code_groupings(md_pct_ranks, cell_metadata):
  rng = np.random.default_rng()
  pointwise_groupings = []
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

    code_range = list(range(code_count))
    for cell_id, md_abs_idx in markdown_ids.items():
      if code_count <= 3: 
        code_vectors = np.asarray(code_range, dtype=np.int32) + doc_start
        pointwise_groupings.append([md_abs_idx, code_vectors])
        labels.append(md_pct_ranks[cell_id])
      else:
        code_vectors = generate_code_vectors(code_range, rng) + doc_start
        for code_vector in code_vectors:
          pointwise_groupings.append([md_abs_idx, code_vector])
          labels.append(md_pct_ranks[cell_id])
    doc_start += doc_length
  labels = np.asarray(labels, dtype=np.float32)
  return pointwise_groupings, labels

def generate_code_vectors(code_range, rng):
  """Generates code vectors depending on size"""
  code_vectors = []
  code_count = len(code_range)
  quarter_loc, remainder = divmod(code_count, 4)
  drop_remainder = remainder
  
  # Adjust samples per quarter
  if code_count <= 13:  # From 10, select 4
    samples_per_quarter = 1
    max_combinations = 1
  elif code_count <= 18:  # From 18, select 8
    samples_per_quarter = 2
    max_combinations = 2
  else:  # From 18+, select 12
    samples_per_quarter = 3
    max_combinations = 3

  if remainder != 0:
    quarter_combinations = np.asarray(list(itertools.combinations(range(quarter_loc+1), samples_per_quarter)))
  else:
    quarter_combinations = np.asarray(list(itertools.combinations(range(quarter_loc), samples_per_quarter)))

  for i in range(4):
    sampled_quarter = rng.choice(quarter_combinations, max_combinations, replace=False)
    if drop_remainder != 0:  # Tick down drop_remainder
      drop_remainder -= 1
      if drop_remainder == 0:
        # Create new combinations for the last one that includes the shifted remainder
        quarter_combinations = np.asarray(list(itertools.combinations(range(quarter_loc), samples_per_quarter)))
        quarter_combinations += quarter_loc*i + remainder
      else:
        quarter_combinations += quarter_loc + 1  # Remainder is present, shift by 1 for the quadrants as you consume the remainder
    else:
      quarter_combinations += quarter_loc  # No remainder, just add the quarter loc to the index
    code_vectors.append(sampled_quarter)
  code_vectors = np.stack(code_vectors, axis=1).astype(np.int32)
  return code_vectors 

def create_pointwise_features(input_ids,
                              pointwise_groupings,
                              tokenizer,
                              sequence_length):
  feature_input_ids = []
  feature_attention_masks = []
  code_split_token = tokenizer('<c>')['input_ids'][1]
  
  for md_idx, code_grouping in tqdm(pointwise_groupings, desc='Assembling Input Ids and Attention Masks'):
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
      code_cells = np.take(input_ids, code_grouping)
      code_cells = np.concatenate(code_cells, axis=0)
      code_input_id =  list(code_cells) + [code_split_token]  # code_split_token is removed below
    
    input_id = markdown_input_id + code_input_id[:-1] + [tokenizer.sep_token_id]
    attention = [1] * len(input_id)
    no_attention = [0] * (sequence_length - len(attention))
    attention_mask = attention + no_attention
    
    input_id_padding = [tokenizer.pad_token_id] * (sequence_length - len(input_id))
    input_id = input_id + input_id_padding
    feature_input_ids.append(input_id)
    feature_attention_masks.append(attention_mask)   

  feature_input_ids = np.asarray(feature_input_ids, dtype=np.int32)
  feature_attention_masks = np.asarray(feature_attention_masks, dtype=np.int32)      
  return [feature_input_ids, feature_attention_masks]

def sample_and_shuffle(features, labels, samples_per_file, random_state):
  rng = np.random.default_rng(random_state)
  shuffler = np.arange(len(labels))
  rng.shuffle(shuffler)

  features[0] = np.take(features[0], shuffler, axis=0)[:samples_per_file]
  features[1] = np.take(features[1], shuffler, axis=0)[:samples_per_file]
  labels = np.take(labels, shuffler, axis=0)[:samples_per_file]
  return tuple(features), labels