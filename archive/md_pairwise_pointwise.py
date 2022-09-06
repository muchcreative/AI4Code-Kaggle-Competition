"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import copy
import random

import itertools
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def get_md_pct_ranks(cell_metadata, orders):
  md_pct_ranks = {}
  doc_start = 0

  for metadata in cell_metadata:
    doc_length = len(metadata)
    doc_end = doc_start + doc_length
    doc_orders = orders[doc_start:doc_end]

    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'markdown':
        md_pct_ranks[cell_id] = doc_orders[idx]/doc_length
    doc_start += doc_length
  return md_pct_ranks

def collect_pointwise_groupings(md_pct_ranks, cell_metadata):
  doc_start = 0
  pointwise_groupings = []
  labels = []

  for count, metadata in enumerate(tqdm(cell_metadata, desc='Collecting Markdown Pairings')):
    doc_length = len(metadata)
    doc_end = doc_start + doc_length
    markdown_ids = {}
    code_count = 0
    md_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        markdown_ids[cell_id] = idx+doc_start
        md_count += 1

    code_samples = list(range(code_count))
    for cell_id, abs_idx in markdown_ids.items():
      if len(code_samples) <= 3:  # Less than 3 code cells, can't split into quadrants
        sampled_code_vector = np.asarray(code_samples, dtype=np.int32) + doc_start
      else:
        sampled_code_vector = create_code_vector(code_samples) + doc_start
      pointwise_groupings.append([abs_idx, sampled_code_vector])
      labels.append(md_pct_ranks[cell_id])
    
    doc_start += doc_length
  labels = np.asarray(labels, dtype=np.float32)
  return pointwise_groupings, labels

def create_code_vector(code_samples):
  """Can try forcing to take first and last code cells, might improve accuracy."""
  code_vector = []
  quarter_loc = int(len(code_samples)/4)

  if quarter_loc >= 3:
    samples_per_quarter = 3
  else:
    samples_per_quarter = quarter_loc

  for i in range(4):
    if i == 3:  # Final quarter, take slice to the end of the index
      sampled_quarter = np.random.choice(code_samples[quarter_loc*i:], samples_per_quarter, replace=False)
    else:
      sampled_quarter = np.random.choice(code_samples[quarter_loc*i:quarter_loc*(i+1)], samples_per_quarter, replace=False)
    sampled_quarter.sort()
    code_vector.append(sampled_quarter)
  
  code_vector = np.asarray(code_vector, dtype=np.int32)
  return code_vector 

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
    for code_quadrant in code_grouping:
      print(md_idx)
      print(code_grouping)
      quadrant = np.take(input_ids, code_quadrant)
      quadrant = np.concatenate(quadrant, axis=0)
      code_input_id +=  list(quadrant) + [code_split_token]

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

def collect_pairwise_groupings(text,
                               orders,
                               cell_metadata,
                               code_cell_count,
                               samples_per_doc):
  pairwise_groupings = []
  doc_start = 0

  for count, metadata in enumerate(tqdm(cell_metadata, desc='Collecting Markdown Pairings')):
    doc_length = len(metadata)
    doc_end = doc_start + doc_length
    code_count = 0
    md_count = 0
    
    doc_orders = orders[doc_start:doc_end]
    for cell_type in metadata.values():
      if cell_type == 'code':
        code_count += 1
      else:
        md_count += 1

    md_pairs = list(itertools.permutations(range(md_count), 2))
    random.shuffle(md_pairs)
    md_pairs = md_pairs[:samples_per_doc]

    code_samples = list(range(code_count))
    code_vector = create_code_vector(code_samples)
    doc_start += doc_length
  return pairwise_groupings

def sample_and_shuffle(features, labels, samples_per_file, random_state):
  rng = np.random.default_rng(random_state)
  shuffler = np.arange(len(labels))
  rng.shuffle(shuffler)

  features[0] = np.take(features[0], shuffler, axis=0)[:samples_per_file]
  features[1] = np.take(features[1], shuffler, axis=0)[:samples_per_file]
  labels = np.take(labels, shuffler, axis=0)[:samples_per_file]
  return tuple(features), labels

# def collect_markdown_groupings(text, df_orders, cell_metadata, doc_ids):
#   all_cell_ids = np.asarray([cell_id for cell_ids in cell_metadata for cell_id in cell_ids])
#   markdown_groupings = []

#   for count, doc_id in enumerate(tqdm(doc_ids, desc='Collecting Markdown Groupings')):
#     prev_cell_type = None
#     prev_cell_id = None
#     md_group = []
#     for cell_id in df_orders[doc_id]:
#       cell_type = cell_metadata[count][cell_id]

#       if cell_type == 'markdown':
#         if prev_cell_type == 'markdown':
#           if md_group:
#             md_group.append(get_text_from_cell_id(cell_id, all_cell_ids, text))
#           else:
#             md_group.extend([get_text_from_cell_id(prev_cell_id, all_cell_ids, text),
#                              get_text_from_cell_id(cell_id, all_cell_ids, text)])
#         prev_cell_type = 'markdown'
#         prev_cell_id = cell_id

#       else:  # cell_type is a code cell
#         if md_group:
#           markdown_groupings.append(md_group)
#           md_group = []
#         prev_cell_type = 'code'
#         prev_cell_id = cell_id
#   return markdown_groupings

# def get_text_from_cell_id(target_cell_id, cell_ids, text):
#   cell_id_loc = np.nonzero(cell_ids == target_cell_id)[0][0]
#   return text[cell_id_loc]

# def shuffle_markdown_groups(ordered_markdown_groups):
#   shuffled_groups = []
#   shuffled_indices = []

#   for md_group in tqdm(ordered_markdown_groups, desc='Shuffing Markdown Groups'):
#     shuffler = list(range(len(md_group)))
#     random.shuffle(shuffler)

#     shuffled_groups.append([md_group[shuffled_idx] for shuffled_idx in shuffler])
#     shuffler = np.asarray(shuffler, dtype=np.float32) + 1
#     shuffled_indices.append(shuffler)
#   return shuffled_groups, shuffled_indices

# def create_features(md_groups, tokenizer, sequence_length):
#   all_input_ids = []
#   all_attention_masks = []
#   for md_group in tqdm(md_groups, desc='Creating Features'):
#     input_ids = []
#     max_length = int(round(sequence_length/len(md_group))) - 2  # include CLS + SEP token
#     for text in md_group:
#       input_ids.extend(tokenize(text, tokenizer, max_length))
    
#     input_ids, attention_mask = add_padding_and_attention(input_ids, tokenizer, sequence_length)
#     all_input_ids.append(input_ids)
#     all_attention_masks.append(attention_mask)

#   all_input_ids = np.asarray(all_input_ids, dtype=np.int32)
#   all_attention_masks = np.asarray(all_attention_masks, dtype=np.int32)
#   features = [all_input_ids, all_attention_masks]
#   return features

# def tokenize(text, tokenizer, max_length):
#   input_id = tokenizer(text,
#                        max_length=max_length,
#                        padding=False,
#                        truncation=True,
#                        add_special_tokens=True,
#                        return_attention_mask=False)['input_ids']
#   return input_id

# def add_padding_and_attention(input_ids, tokenizer, sequence_length):
#   attention = [1] * len(input_ids)
#   no_attention = [0] * (sequence_length - len(attention))
#   attention_mask = attention + no_attention
#   input_id_padding = [tokenizer.pad_token_id] * (sequence_length - len(input_ids))
#   input_ids = input_ids + input_id_padding
#   return input_ids, attention_mask

# def create_labels(shuffled_indices, max_width):
#   labels = []
#   for shuffled_index in shuffled_indices:
#     req_pad = max_width - len(shuffled_index)
#     labels.append(np.pad(shuffled_index, (0, req_pad), 'constant'))
#   labels = np.asarray(labels, dtype=np.float32)
#   return labels