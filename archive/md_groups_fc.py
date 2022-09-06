"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import copy
import random
import itertools
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def collect_markdown_groupings(text, df_orders, cell_metadata, doc_ids):
  all_cell_ids = np.asarray([cell_id for cell_ids in cell_metadata for cell_id in cell_ids])
  markdown_groupings = []

  for count, doc_id in enumerate(tqdm(doc_ids, desc='Collecting Markdown Groupings')):
    prev_cell_type = None
    prev_cell_id = None
    md_group = []
    for cell_id in df_orders[doc_id]:
      cell_type = cell_metadata[count][cell_id]

      if cell_type == 'markdown':
        if prev_cell_type == 'markdown':
          if md_group:
            md_group.append(get_text_from_cell_id(cell_id, all_cell_ids, text))
          else:
            md_group.extend([get_text_from_cell_id(prev_cell_id, all_cell_ids, text),
                             get_text_from_cell_id(cell_id, all_cell_ids, text)])
        prev_cell_type = 'markdown'
        prev_cell_id = cell_id

      else:  # cell_type is a code cell
        if md_group:
          markdown_groupings.append(md_group)
          md_group = []
        prev_cell_type = 'code'
        prev_cell_id = cell_id
  return markdown_groupings

def get_text_from_cell_id(target_cell_id, cell_ids, text):
  cell_id_loc = np.nonzero(cell_ids == target_cell_id)[0][0]
  return text[cell_id_loc]

def shuffle_markdown_groups(ordered_markdown_groups):
  shuffled_groups = []
  shuffled_indices = []

  for md_group in tqdm(ordered_markdown_groups, desc='Shuffing Markdown Groups'):
    # No need to random shuffle, just create all possible iterations
    # = len(md_group)
    shuffler = list(range(len(md_group)))
    random.shuffle(shuffler)

    shuffled_groups.append([md_group[shuffled_idx] for shuffled_idx in shuffler])
    shuffler = np.asarray(shuffler, dtype=np.float32) + 1
    shuffled_indices.append(shuffler)
  return shuffled_groups, shuffled_indices

def create_features(md_groups, tokenizer, sequence_length):
  all_input_ids = []
  all_attention_masks = []
  for md_group in tqdm(md_groups, desc='Creating Features'):
    input_ids = []
    max_length = int(round(sequence_length/len(md_group))) - 2  # include CLS + SEP token
    for text in md_group:
      input_ids.extend(tokenize(text, tokenizer, max_length))
    
    input_ids, attention_mask = add_padding_and_attention(input_ids, tokenizer, sequence_length)
    all_input_ids.append(input_ids)
    all_attention_masks.append(attention_mask)

  all_input_ids = np.asarray(all_input_ids, dtype=np.int32)
  all_attention_masks = np.asarray(all_attention_masks, dtype=np.int32)
  features = [all_input_ids, all_attention_masks]
  return features

def tokenize(text, tokenizer, max_length):
  input_id = tokenizer(text,
                       max_length=max_length,
                       padding=False,
                       truncation=True,
                       add_special_tokens=True,
                       return_attention_mask=False)['input_ids']
  return input_id

def add_padding_and_attention(input_ids, tokenizer, sequence_length):
  attention = [1] * len(input_ids)
  no_attention = [0] * (sequence_length - len(attention))
  attention_mask = attention + no_attention
  input_id_padding = [tokenizer.pad_token_id] * (sequence_length - len(input_ids))
  input_ids = input_ids + input_id_padding
  return input_ids, attention_mask

def create_labels(shuffled_indices, max_width):
  labels = []
  for shuffled_index in shuffled_indices:
    req_pad = max_width - len(shuffled_index)
    labels.append(np.pad(shuffled_index, (0, req_pad), 'constant'))
  labels = np.asarray(labels, dtype=np.float32)
  return labels

def sample_and_shuffle(features, labels, samples_per_file):
  shuffler = list(range(len(labels)))
  random.shuffle(shuffler)
  features[0] = np.take(features[0], shuffler, axis=0)[:samples_per_file]
  features[1] = np.take(features[1], shuffler, axis=0)[:samples_per_file]
  labels = np.take(labels, shuffler, axis=0)[:samples_per_file]
  return tuple(features), labels