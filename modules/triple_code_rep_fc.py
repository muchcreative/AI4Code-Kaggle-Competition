"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import copy

import pandas as pd
import itertools
import tensorflow as tf

from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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
    assert max(abs_pct_ranks) < 1.0 and min(abs_pct_ranks) > -1.0, 'Max and Min Absolute PCT Rank is (-1, 1) for tanh activation'

    doc_pct_ranks = {md_ids[i]: abs_pct_ranks[i] for i in range(md_count)}
    md_pct_ranks.update(doc_pct_ranks)
  return md_pct_ranks

def create_quadrant_splits(code_cell_count):
  quarter_loc, remainder = divmod(code_cell_count, 4)
  quadrant_splits = np.arange(0, code_cell_count-remainder+0.1, step=quarter_loc, dtype=np.int32)
  for i in range(0, remainder):
    quadrant_splits[i+1:] = quadrant_splits[i+1:] + 1
  return quadrant_splits

def collect_md_code_groupings(c1_input_ids,
                              c2_input_ids,
                              c3_input_ids,
                              cell_metadata,
                              md_pct_ranks):
  c1_groupings = []
  c2_groupings = []
  c3_groupings = []
  labels = []

  main_len = 70
  multi_len = 130
  doc_start = 0

  for count, metadata in enumerate(tqdm(cell_metadata, desc='Collecting Markdown Code Groupings')):
    doc_len = len(metadata)
    md_ids = []
    code_count = 0
    md_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        md_ids.append(cell_id)
        md_count += 1

    main_md_idxs = list(range(md_count))
    vec_md_idxs = list(range(md_count))
    quarter_loc = code_count // 4

    if code_count >= 4:
      quadrant_splits = create_quadrant_splits(code_count)

    main_c1_input_ids = adjust_input_ids_len(c1_input_ids[doc_start+code_count:doc_start+doc_len], main_len)
    main_c2_input_ids = adjust_input_ids_len(c2_input_ids[doc_start+code_count:doc_start+doc_len], main_len)
    main_c3_input_ids = adjust_input_ids_len(c3_input_ids[doc_start+code_count:doc_start+doc_len], multi_len)

    sub_c1_input_ids = c1_input_ids[doc_start:doc_start+code_count]
    sub_c2_input_ids = c2_input_ids[doc_start:doc_start+code_count]
    sub_c3_input_ids = c3_input_ids[doc_start:doc_start+code_count]

    # Removes duplicates as the main markdown
    main_md_idxs, _ = remove_duplicates(main_md_idxs, main_c1_input_ids)  # Only need one to reference

    # Create random vectors for all occurances
    for md_idx in main_md_idxs:
      main_c1_input_id = main_c1_input_ids[md_idx]
      main_c2_input_id = main_c2_input_ids[md_idx]
      main_c3_input_id = main_c3_input_ids[md_idx]

      if code_count >= 4:
        code_rep_idxs = create_doc_code_rep(quarter_loc, quadrant_splits)
        code_rep1 = map_code_rep_to_input_ids(code_rep_idxs[0], sub_c1_input_ids, main_len)
        code_rep2 = map_code_rep_to_input_ids(code_rep_idxs[1], sub_c2_input_ids, main_len)
        code_rep3 = map_code_rep_to_input_ids(code_rep_idxs[2], sub_c3_input_ids, multi_len)
      else:
        code_rep_idxs = np.arange(code_count)
        code_rep1 = map_single_rep_to_input_ids(code_rep_idxs, sub_c1_input_ids, main_len)
        code_rep2 = map_single_rep_to_input_ids(code_rep_idxs, sub_c2_input_ids, main_len)
        code_rep3 = map_single_rep_to_input_ids(code_rep_idxs, sub_c3_input_ids, multi_len)
        
      c1_groupings.append([main_c1_input_id, code_rep1])
      c2_groupings.append([main_c2_input_id, code_rep2])
      c3_groupings.append([main_c3_input_id, code_rep3])
      labels.append(md_pct_ranks[md_ids[md_idx]])

    doc_start += doc_len
  labels = np.asarray(labels, dtype=np.float32)
  labels = np.reshape(labels, (len(labels), 1))
  return c1_groupings, c2_groupings, c3_groupings, labels

def adjust_input_ids_len(input_ids, max_len):
  halfway_len, remainder = divmod(max_len, 2)
  adjusted_input_ids = []
  for input_id in input_ids:
    if len(input_id) > max_len:
      adjusted_input_ids.append(input_id[:halfway_len+remainder] + input_id[-halfway_len:])
      assert len(adjusted_input_ids[-1]) == max_len, 'You summed the input ids, please double check concatenation.'
    else:
      adjusted_input_ids.append(input_id)
  return adjusted_input_ids

def remove_duplicates(main_md_idxs, main_md_input_ids):
  seen_mds = {}
  dupe_mds = []

  for md_idx, main_md_ii in enumerate(main_md_input_ids):
    if tuple(main_md_ii) not in seen_mds:
      seen_mds[tuple(main_md_ii)] = md_idx
    else:
      dupe_mds.extend([seen_mds[tuple(main_md_ii)], md_idx])
  
  dupe_mds = sorted(set(dupe_mds))[::-1]
  for dupe_md in dupe_mds:
    del main_md_idxs[dupe_md]
  return main_md_idxs, dupe_mds

def create_doc_code_rep(quarter_loc, quadrant_splits):
  if quarter_loc <= 4:
    code_rep = [] 
    for i in range(4):
      quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1])
      code_rep.append(quadrant_vector)
    return [code_rep] * 3

  else:
    first_rep = []
    second_rep = []
    third_rep = []

    for i in range(4):
      # Quad Rep 1
      quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1])
      spacing = (len(quadrant_vector)-2) // 3
      quad_1_idxs = [0, spacing, spacing*2, spacing*3, -1]
      quad_rep_1 = quadrant_vector[quad_1_idxs]

      # Quad Rep 2
      if len(quadrant_vector) == 6:
        quad_2_idxs = [0, 1, 2, 3, 5]
      elif len(quadrant_vector) == 7:
        quad_2_idxs = [1, 2, 3, 4, 5]
      elif len(quadrant_vector) >= 8:
        reduced_vector = np.arange(quadrant_splits[i]+1, quadrant_splits[i+1]-1)
        spacing, remainder = divmod(len(reduced_vector)-2, 3)
        quad_2_idxs = [1, spacing+remainder, spacing*2+remainder, spacing*3+remainder, -2]
      else:
        quad_2_idxs = quad_1_idxs
      quad_rep_2 = quadrant_vector[quad_2_idxs]

      # Quad Rep 3
      spacing, remainder = divmod(len(quadrant_vector), 3)
      quad_3_idxs = [0, spacing + remainder//2, spacing*2 + remainder//2, -1]
      quad_rep_3 = quadrant_vector[quad_3_idxs]

      first_rep.append(quad_rep_1)
      second_rep.append(quad_rep_2)
      third_rep.append(quad_rep_3)
    return [first_rep, second_rep, third_rep]
    
def map_single_rep_to_input_ids(code_rep, input_ids, main_len):
  avaliable_length = 512 - main_len - 7
  code_len = int(avaliable_length / len(code_rep))
  halfway_len, remainder = divmod(code_len, 2)

  mapped_rep = []
  carry_over = 0

  for idx in code_rep:
    current_ii = input_ids[idx]
    if len(current_ii) > code_len + carry_over:
      req_carry_over = min(len(current_ii) - code_len, carry_over)
      carry_len, carry_r = divmod(req_carry_over, 2)

      mapped_rep.append(current_ii[:halfway_len+remainder+carry_len] + current_ii[-halfway_len-carry_len-carry_r:])
      carry_over -= req_carry_over
      assert len(mapped_rep[-1]) == code_len+req_carry_over, 'You summed the input ids, please double check concatenation.'
    else:
      mapped_rep.append(current_ii)
      carry_over += code_len - len(current_ii)
  return mapped_rep

def map_code_rep_to_input_ids(code_rep, input_ids, md_len):
  avaliable_length = 512 - md_len - 7
  code_len = int(avaliable_length / len(np.concatenate(code_rep)))
  halfway_len, remainder = divmod(code_len, 2)

  mapped_rep = []
  carry_over = 0

  for quadrant in code_rep:
    quadrant_rep = []
    for idx in quadrant:
      current_ii = input_ids[idx]
      if len(current_ii) > code_len + carry_over:
        req_carry_over = min(len(current_ii) - code_len, carry_over)
        carry_len, carry_r = divmod(req_carry_over, 2)

        quadrant_rep.append(current_ii[:halfway_len+remainder+carry_len] + current_ii[-halfway_len-carry_len-carry_r:])
        carry_over -= req_carry_over
        assert len(quadrant_rep[-1]) == (code_len+req_carry_over), 'You summed the input ids, please double check concatenation.'
      else:
        quadrant_rep.append(current_ii)
        carry_over += code_len - len(current_ii)
  
    quadrant_rep = np.concatenate(quadrant_rep, dtype=np.int32)
    mapped_rep.append(quadrant_rep)
  return mapped_rep

def create_features(groupings, tokenizer, md_len, disable_print=False):
  feature_input_ids = []
  feature_attention_masks = []
  
  doc_split_token = tokenizer('<d>')['input_ids'][1]
  group_split_token = tokenizer('<c>')['input_ids'][1]
  seq_len = 512

  for md_input_id, grouping in tqdm(groupings, desc='Assembling Input Ids and Attention Masks', disable=disable_print):
    req_padding = md_len - len(md_input_id)
    
    md_am = [1] + [1]*len(md_input_id) + [0]*req_padding + [1]*2  # CLS + md_input_id + padding + SEP + DOC
    markdown_input_id = [tokenizer.cls_token_id] + md_input_id + [0]*req_padding + [tokenizer.sep_token_id] + [doc_split_token]
  
    group_input_id = []
    for group in grouping:
      group_input_id +=  list(group) + [group_split_token]

    input_id = markdown_input_id + group_input_id[:-1] + [tokenizer.sep_token_id]
    input_am = md_am + [1] * len(group_input_id)
    padding_am = [0] * (seq_len - len(input_am))
    am = input_am + padding_am
    
    input_id_padding = [tokenizer.pad_token_id] * (seq_len - len(input_id))
    input_id = input_id + input_id_padding

    feature_input_ids.append(input_id)
    feature_attention_masks.append(am) 

  feature_input_ids = np.asarray(feature_input_ids, dtype=np.int32)
  feature_attention_masks = np.asarray(feature_attention_masks, dtype=np.int32)      
  return [feature_input_ids, feature_attention_masks]

def collect_best_code_groupings(c1_input_ids,
                                c2_input_ids,
                                c3_input_ids,
                                cell_metadata):
  c1_groupings = []
  c2_groupings = []
  c3_groupings = []

  main_len = 70
  multi_len = 130
  doc_start = 0

  for count, metadata in enumerate(cell_metadata):
    doc_len = len(metadata)
    md_ids = []
    code_count = 0
    md_count = 0
 
    for idx, cell_id in enumerate(metadata):
      if metadata[cell_id] == 'code':
        code_count += 1
      else:
        md_ids.append(cell_id)
        md_count += 1

    main_md_idxs = list(range(md_count))
    vec_md_idxs = list(range(md_count))
    quarter_loc = code_count // 4

    if code_count >= 4:
      quadrant_splits = create_quadrant_splits(code_count)

    main_c1_input_ids = adjust_input_ids_len(c1_input_ids[doc_start+code_count:doc_start+doc_len], main_len)
    main_c2_input_ids = adjust_input_ids_len(c2_input_ids[doc_start+code_count:doc_start+doc_len], main_len)
    main_c3_input_ids = adjust_input_ids_len(c3_input_ids[doc_start+code_count:doc_start+doc_len], multi_len)

    sub_c1_input_ids = c1_input_ids[doc_start:doc_start+code_count]
    sub_c2_input_ids = c2_input_ids[doc_start:doc_start+code_count]
    sub_c3_input_ids = c3_input_ids[doc_start:doc_start+code_count]

    # Removes duplicates as the main markdown
    main_md_idxs, md_dupes = remove_duplicates(main_md_idxs, main_c1_input_ids)  # Only need one to reference

    # Create random vectors for all occurances
    for md_idx in main_md_idxs:
      main_c1_input_id = main_c1_input_ids[md_idx]
      main_c2_input_id = main_c2_input_ids[md_idx]
      main_c3_input_id = main_c3_input_ids[md_idx]

      if code_count >= 4:
        code_rep_idxs = create_doc_code_rep(quarter_loc, quadrant_splits)
        code_rep1 = map_code_rep_to_input_ids(code_rep_idxs[0], sub_c1_input_ids, main_len)
        code_rep2 = map_code_rep_to_input_ids(code_rep_idxs[1], sub_c2_input_ids, main_len)
        code_rep3 = map_code_rep_to_input_ids(code_rep_idxs[2], sub_c3_input_ids, multi_len)
      else:
        code_rep_idxs = np.arange(code_count)
        code_rep1 = map_single_rep_to_input_ids(code_rep_idxs, sub_c1_input_ids, main_len)
        code_rep2 = map_single_rep_to_input_ids(code_rep_idxs, sub_c2_input_ids, main_len)
        code_rep3 = map_single_rep_to_input_ids(code_rep_idxs, sub_c3_input_ids, multi_len)
        
      c1_groupings.append([main_c1_input_id, code_rep1])
      c2_groupings.append([main_c2_input_id, code_rep2])
      c3_groupings.append([main_c3_input_id, code_rep3])

    doc_start += doc_len
  return c1_groupings, c2_groupings, c3_groupings, md_dupes