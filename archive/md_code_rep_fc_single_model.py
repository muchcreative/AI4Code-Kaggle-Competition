"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import copy

import pandas as pd
import itertools
import tensorflow as tf

from tqdm.auto import tqdm

def get_md_pct_ranks(doc_ids, cell_metadata, df_orders):
  md_pct_ranks = {}

  for count, doc_id in enumerate(doc_ids):
    metadata = cell_metadata[count]
    code_count = list(metadata.values()).count('code')
    md_count = list(metadata.values()).count('markdown')
    doc_orders = df_orders[doc_id]

    md_memory = []
    md_ids = []
    rel_pct_ranks = []
    current_code_idx = 0

    for cell_id in doc_orders:
      if metadata[cell_id] == 'markdown':
        md_memory.append(cell_id)
        md_ids.append(cell_id)
      else:  # cell_id is a code cell
        if md_memory:
          for idx, md_id in enumerate(md_memory, start=1):
            rel_pct_ranks.append(current_code_idx + idx*(1/(len(md_memory)+1)))
        current_code_idx += 1
        md_memory = []
    
    # For last markdown idxs
    if md_memory:
      for idx, md_id in enumerate(md_memory, start=1):
        rel_pct_ranks.append(current_code_idx + idx*(1/(len(md_memory)+1)))
    abs_pct_ranks = np.asarray(rel_pct_ranks, dtype=np.float32) / (code_count+1)
    abs_pct_ranks = (abs_pct_ranks * 2) - 1  # Rescales to (1, -1)
    assert max(abs_pct_ranks) < 1.0 and min(abs_pct_ranks) > -1.0, 'Max and Min Absolute PCT Rank is [-1, 1] for tanh activation'
    
    doc_pct_ranks = {md_ids[i]: abs_pct_ranks[i] for i in range(md_count)}
    md_pct_ranks.update(doc_pct_ranks)
  return md_pct_ranks

def create_quadrant_splits(code_cell_count):
  quarter_loc, remainder = divmod(code_cell_count, 4)
  quadrant_splits = np.arange(0, code_cell_count-remainder+0.1, step=quarter_loc, dtype=np.int32)
  for i in range(0, remainder):
    quadrant_splits[i+1:] = quadrant_splits[i+1:] + 1
  return quadrant_splits

def collect_md_code_groupings(input_ids, cell_metadata, md_pct_ranks):
  md_groupings = []
  md_quad_groupings = []
  labels = []

  main_md_len = 81
  main_md_quad_len = 66

  md_len = 38
  code_len = 22
  doc_start = 0
  rng = np.random.default_rng()

  for metadata in tqdm(cell_metadata, desc='Collecting Markdown Code Groupings'):
    doc_len = len(metadata)
    md_ids = {}
    code_count = 0
    md_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        md_ids[cell_id] = idx - code_count  # First markdown is at index 0
        md_count += 1

    md_idxs = list(md_ids.values())
    quarter_loc, _ = divmod(code_count, 4)

    if code_count >= 4:
      quadrant_splits = create_quadrant_splits(code_count)

    main_md_input_ids = adjust_input_ids_len(input_ids[doc_start+code_count:doc_start+doc_len], main_md_len)
    md_input_ids = adjust_input_ids_len(input_ids[doc_start+code_count:doc_start+doc_len], md_len)

    main_md_quad_input_ids = adjust_input_ids_len(input_ids[doc_start+code_count:doc_start+doc_len], main_md_quad_len)
    code_input_ids = adjust_input_ids_len(input_ids[doc_start:doc_start+code_count], code_len)

    # Create random vectors for all occurances
    for md_id, md_idx in md_ids.items():
      main_md_input_id = main_md_input_ids[md_idx]
      main_md_quad_input_id = main_md_quad_input_ids[md_idx]
      md_rep = create_doc_md_rep(md_idx, md_idxs, rng)
      if code_count >= 4:
        code_rep = create_doc_code_rep(quarter_loc, quadrant_splits, rng)
      else:
        code_rep = [np.arange(code_count)]

      # Truncate excess representations if it doesn't have a partner                
      while len(md_rep) != len(code_rep):
        if len(md_rep) > len(code_rep):
          md_rep.pop()
        else:
          code_rep.pop()

      md_rep = map_md_rep_to_input_ids(md_rep, md_input_ids)
      if code_count >= 4:
        code_rep = map_code_rep_to_input_ids(code_rep, code_input_ids)
      else:
        code_rep = map_md_rep_to_input_ids(code_rep, code_input_ids)
        
      for i in range(len(md_rep)):
        md_groupings.append([main_md_input_id, md_rep[i]])
        md_quad_groupings.append([main_md_quad_input_id, code_rep[i]])
        labels.append(md_pct_ranks[md_id])
    doc_start += doc_len
  labels = np.asarray(labels, dtype=np.float32)
  labels = np.reshape(labels, (len(labels), 1))
  return md_groupings, md_quad_groupings, labels

def adjust_input_ids_len(input_ids, max_len):
  halfway_len, remainder = divmod(max_len, 2)
  adjusted_input_ids = []
  for input_id in input_ids:
    if len(input_id) > max_len:
      adjusted_input_ids.append(input_id[:halfway_len+remainder] + input_id[-halfway_len:])
      assert len(adjusted_input_ids[-1]) != halfway_len, 'You summed the input ids, please double check concatenation.'
    else:
      adjusted_input_ids.append(input_id)
  return adjusted_input_ids

def create_doc_md_rep(md_idx, md_idxs, rng):
  md_rep = []
  excluded_md_idxs = md_idxs[:]
  excluded_md_idxs.remove(md_idx)
  md_count = len(md_idxs)

  if md_count == 2:
    md_rep = [excluded_md_idxs]
  elif md_count <= 11:
    for i in range(2):
      rng.shuffle(excluded_md_idxs)
      md_rep.append(excluded_md_idxs[:11])
  elif md_count <= 16:
    for i in range(3):
      rng.shuffle(excluded_md_idxs)
      md_rep.append(excluded_md_idxs[:11])
  else:
    for i in range(4):
      rng.shuffle(excluded_md_idxs)
      md_rep.append(excluded_md_idxs[:11])
  return md_rep

def create_doc_code_rep(quarter_loc, quadrant_splits, rng):
  """Code count must always be equal to or greater than 4
  doc_start is for creating features all at once instead of per document id
  """
  code_rep = []

  if quarter_loc <= 4:
    for i in range(4):
      quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1])
      code_rep.append(quadrant_vector)
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
    code_rep = list(np.stack(quadrant_vectors, axis=1))
  return code_rep

def map_md_rep_to_input_ids(reps, input_ids):
  mapped_rep = []
  for rep in reps:
    mapped_rep.append([input_ids[idx] for idx in rep])
  return mapped_rep

def map_code_rep_to_input_ids(reps, input_ids):
  mapped_rep = []
  for rep in reps:
    code_vector = []
    for quad in rep:
      quad_input_ids = [input_ids[idx] for idx in quad]
      quad_input_ids = np.concatenate(quad_input_ids)
      code_vector.append(quad_input_ids)
    mapped_rep.append(code_vector)
  return mapped_rep

def create_features(groupings, tokenizer, disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []
  code_split_token = tokenizer('<c>')['input_ids'][1]
  seq_len = 512

  for md_input_id, grouping in tqdm(groupings, desc='Assembling Input Ids and Attention Masks', disable=disable_print):
    code_input_id = []
    markdown_input_id = ([tokenizer.cls_token_id]
                          + md_input_id
                          + [tokenizer.sep_token_id])
                          
    for group in grouping:
      code_input_id +=  list(group) + [code_split_token]

    input_id = markdown_input_id + code_input_id[:-1] + [tokenizer.eos_token_id]
    attention = [1] * len(input_id)
    no_attention = [0] * (seq_len - len(attention))
    attention_mask = attention + no_attention
    
    input_id_padding = [tokenizer.pad_token_id] * (seq_len - len(input_id))
    input_id = input_id + input_id_padding
    if len(input_id) != 512:
      print(len(input_id))
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

def collect_md_code_groupings_no_labels(input_ids, cell_metadata):
  md_groupings = []
  md_quad_groupings = []

  main_md_len = 81
  main_md_quad_len = 66

  md_len = 38
  code_len = 22
  doc_start = 0
  rng = np.random.default_rng()

  for metadata in cell_metadata:
    doc_len = len(metadata)
    md_ids = {}
    code_count = 0
    md_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        md_ids[cell_id] = idx - code_count  # First markdown is at index 0
        md_count += 1

    md_idxs = list(md_ids.values())
    quarter_loc, _ = divmod(code_count, 4)

    if code_count >= 4:
      quadrant_splits = create_quadrant_splits(code_count)

    main_md_input_ids = adjust_input_ids_len(input_ids[doc_start+code_count:doc_start+doc_len], main_md_len)
    md_input_ids = adjust_input_ids_len(input_ids[doc_start+code_count:doc_start+doc_len], md_len)

    main_md_quad_input_ids = adjust_input_ids_len(input_ids[doc_start+code_count:doc_start+doc_len], main_md_quad_len)
    code_input_ids = adjust_input_ids_len(input_ids[doc_start:doc_start+code_count], code_len)

    # Create random vectors for all occurances
    for md_id, md_idx in md_ids.items():
      main_md_input_id = main_md_input_ids[md_idx]
      main_md_quad_input_id = main_md_quad_input_ids[md_idx]
      md_rep = create_doc_md_rep(md_idx, md_idxs, rng)
      if code_count >= 4:
        code_rep = create_doc_code_rep(quarter_loc, quadrant_splits, rng)
      else:
        code_rep = [np.arange(code_count)]

      # Truncate excess representations if it doesn't have a partner                
      while len(md_rep) != len(code_rep):
        if len(md_rep) > len(code_rep):
          md_rep.pop()
        else:
          code_rep.pop()

      md_rep = map_md_rep_to_input_ids(md_rep, md_input_ids)
      if code_count >= 4:
        code_rep = map_code_rep_to_input_ids(code_rep, code_input_ids)
      else:
        code_rep = map_md_rep_to_input_ids(code_rep, code_input_ids)
        
      for i in range(len(md_rep)):
        md_groupings.append([main_md_input_id, md_rep[i]])
        md_quad_groupings.append([main_md_quad_input_id, code_rep[i]])
    doc_start += doc_len

  pred_count_per_md = len(md_rep)  # Fixed for the document
  return md_groupings, md_quad_groupings,  pred_count_per_md