"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np

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
    
def collect_md_code_groupings(md_quadrant_locs, input_ids, cell_metadata):
  md_code_groupings = []
  labels = []
  md_len = 90
  halfway_md = 45
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

    quarter_loc, _ = divmod(code_count, 4)
    quadrant_splits = create_quadrant_splits(code_count)
    code_vector = create_doc_code_vector(quarter_loc,
                                         quadrant_splits,
                                         input_ids,
                                         doc_start)
    for cell_id, md_abs_idx in markdown_ids.items():
      md_input_id = input_ids[md_abs_idx]
      if len(md_input_id) > md_len:
        md_input_id = md_input_id[:halfway_md] + md_input_id[-halfway_md:]
        assert len(md_input_id) == md_len, 'You summed the markdowns, please double check concatenation.'
      md_code_groupings.append([md_input_id, code_vector])
      labels.append(md_quadrant_locs[cell_id])
    doc_start += doc_length
  labels = np.asarray(labels, dtype=np.float32)
  return md_code_groupings, labels

def create_doc_code_vector(quarter_loc, quadrant_splits, input_ids, doc_start=0):
  """Code count must always be equal to or greater than 4
  doc_start is for creating features all at once instead of per document id
  """
  code_rep = []
  carry_over = 0 
  quad_len = 104 # At 104 each, md_len is fixed at 90 with 6 special tokens

  for i in range(4):
    quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1]) + doc_start
    quadrant_vector = np.take(input_ids, quadrant_vector, axis=0)
    quadrant_vector = np.concatenate(quadrant_vector, axis=0)

    if len(quadrant_vector) > (quad_len + carry_over):
      halfway_len = int((quad_len+carry_over)/2)  # Will round down
      quadrant_vector = np.concatenate([quadrant_vector[:halfway_len], quadrant_vector[-halfway_len:]], axis=0)
      assert len(quadrant_vector) <= (quad_len + carry_over), 'You summed the quadrant vectors, please check concatenation.'
      carry_over = 0
    else:
      carry_over += quad_len - len(quadrant_vector)
    code_rep.append(quadrant_vector)
  return code_rep

def create_quadrant_features(md_quad_groupings,
                             tokenizer,
                             seq_len=512,
                             disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []
  code_split_token = tokenizer('<c>')['input_ids'][1]
  
  for md_input_id, code_grouping in tqdm(md_quad_groupings, desc='Assembling Input Ids and Attention Masks', disable=disable_print):
    code_input_id = []
    markdown_input_id = ([tokenizer.cls_token_id]
                          + md_input_id
                          + [tokenizer.sep_token_id])
    assert len(markdown_input_id) == (2 + len(md_input_id)), 'You summed up the markdown tokens.'
                          
    if len(code_grouping) == 4:  # 4 quadrants avaliable
      for code_quadrant in code_grouping:
        code_input_id +=  list(code_quadrant) + [code_split_token]
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

def sample_balance_shuffle(features, labels):
  rng = np.random.default_rng()
  
  largest_class_idx = np.where(labels[:, 0] == 1)[0]
  largest_class_input_ids = np.take(features[0], largest_class_idx, axis=0)
  largest_class_attention_masks = np.take(features[1], largest_class_idx, axis=0)
  
  sorted_class_sizes = np.sort([sum(labels[:, 1]),
                                sum(labels[:, 2]),
                                sum(labels[:, 3])]).astype(np.int32)
  mid_class_size = sorted_class_sizes[1]  # Take mid
  
  shuffler = np.arange(len(largest_class_idx))
  rng.shuffle(shuffler)
  largest_class_input_ids = np.take(largest_class_input_ids, shuffler, axis=0)[:mid_class_size]
  largest_class_attention_masks = np.take(largest_class_attention_masks, shuffler, axis=0)[:mid_class_size]
  largest_class_labels = np.take(labels, largest_class_idx, axis=0)[:mid_class_size]
  
  features[0] = np.delete(features[0], largest_class_idx, axis=0)
  features[1] = np.delete(features[1], largest_class_idx, axis=0)
  labels = np.delete(labels, largest_class_idx, axis=0)

  features[0] = np.concatenate([features[0], largest_class_input_ids])
  features[1] = np.concatenate([features[1], largest_class_attention_masks])
  labels = np.concatenate([labels, largest_class_labels], axis=0)

  shuffler = np.arange(len(labels))
  rng.shuffle(shuffler)
  features[0] = np.take(features[0], shuffler, axis=0)
  features[1] = np.take(features[1], shuffler, axis=0)
  labels = np.take(labels, shuffler, axis=0)
  return tuple(features), labels

def collect_md_code_groupings_no_labels(input_ids, cell_metadata, disable_print=False):
  md_code_groupings = []
  md_len = 90
  halfway_md = 45
  doc_start = 0

  for metadata in tqdm(cell_metadata, desc='Collecting Markdown Code Groupings', disable=disable_print):
    doc_length = len(metadata)
    markdown_ids = {}
    code_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        markdown_ids[cell_id] = idx+doc_start

    quarter_loc, _ = divmod(code_count, 4)
    quadrant_splits = create_quadrant_splits(code_count)
    code_vector = create_doc_code_vector(quarter_loc,
                                         quadrant_splits,
                                         input_ids,
                                         doc_start)
    for cell_id, md_abs_idx in markdown_ids.items():
      md_input_id = input_ids[md_abs_idx]
      if len(md_input_id) > md_len:
        md_input_id = md_input_id[:halfway_md] + md_input_id[-halfway_md:]
      md_code_groupings.append([md_input_id, code_vector])
    doc_start += doc_length
  return md_code_groupings