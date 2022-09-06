"""Creates features that asks the model if the markdown exists in the given ordered code block."""

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm

def adjust_input_ids_for_exist_blocks(input_ids, cell_metadata):
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
  if code_cell_count <= 6:  # 6 or less code cells per block; 489 seq_len
    code_len = 65
    md_len = 102
  else:  # 7-8 code cells per block; 512 seq_len
    code_len = 50
    md_len = 102
  return code_len, md_len
    
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

def create_idx_map_with_labels(cell_metadata, md_quadrant_locs, disable_print=False): 
    idx_map = []
    code_vectors = []
    labels = []
    doc_start = 0

    for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Creating Index Mapping', disable=disable_print)):
      code_count = list(metadata.values()).count('code')
      quarter_loc, remainder = divmod(code_count, 4)
      quadrant_splits = create_quadrant_splits(code_count)
      doc_code_vector = create_doc_code_vector(quarter_loc, quadrant_splits, doc_start)

      md_idxs = [idx
                 for idx, cell_id in enumerate(metadata)
                 if metadata[cell_id] == 'markdown']
      md_quad_locs = np.asarray([md_quadrant_locs[list(metadata)[md_idx]]
                                 for md_idx in md_idxs], dtype=np.int32)

      for quad_num in range(4):
        quad_range = np.arange(quadrant_splits[quad_num], quadrant_splits[quad_num+1]) + doc_start
        quad_labels = np.where(md_quad_locs==quad_num, 1, 0)

        if len(quad_range) >= 9:
          quad_maps = [[md_idx + doc_start, list(quad_range[:4]) + list(quad_range[-3:]), doc_count]
                        for md_idx in md_idxs]
        else:
          quad_maps = [[md_idx + doc_start, list(quad_range), doc_count]
                        for md_idx in md_idxs]

        for quad_map in quad_maps:
          idx_map.append(quad_map)
        labels.append(quad_labels)
      
      code_vectors.append(doc_code_vector)
      doc_start += len(metadata)
    labels = np.concatenate(labels, dtype=np.float32)
    return idx_map, code_vectors, labels

def create_doc_code_vector(quarter_loc, quadrant_splits, doc_start=0):
  """Code count must always be equal to or greater than 4
  doc_start is for creating features all at once instead of per document id
  """
  code_rep = []

  if quarter_loc <= 4:
    for i in range(4):
      quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1])
      code_rep.append(quadrant_vector + doc_start)
    
  else:
    for i in range(4):
      quadrant_vector = np.arange(quadrant_splits[i], quadrant_splits[i+1])
      start_vector = quadrant_vector[:3]
      end_vector = quadrant_vector[-2:]
      code_rep.append(np.concatenate([start_vector, end_vector], axis=0) + doc_start)
  return code_rep

def map_code_vector_input_ids(input_ids, code_vectors, tokenizer):
  """Code vectors will still require the correct markdown.
  They will also need padding and attention masks.
  """

  code_split_token = tokenizer('<c>')['input_ids'][1]
  code_vector_input_ids = []

  for code_vector in code_vectors:
    code_vector_input_id = []
    for quad_idx in code_vector:
      quad_input = np.take(input_ids, quad_idx, axis=0)
      start_quad_input = np.concatenate(quad_input[:3], axis=0)
      end_quad_input = np.concatenate(quad_input[-2:], axis=0)
      quad_input = adjust_quadrant_input_length(start_quad_input, end_quad_input)

      code_vector_input_id += list(quad_input) + [code_split_token]
    code_vector_input_ids.append(code_vector_input_id[:-1])  # Remove last code split token
  return code_vector_input_ids

def adjust_quadrant_input_length(start_quad_input, end_quad_input):
  quad_len = 50  # This is half length, each quadrant has 100 tokens in it
  one_side_quad_len = int(quad_len/2)
  if len(start_quad_input) > quad_len:
    start_quad_input = np.concatenate([start_quad_input[:one_side_quad_len], start_quad_input[-one_side_quad_len:]], axis=0)
  if len(end_quad_input) > quad_len:
    end_quad_input =  np.concatenate([end_quad_input[:one_side_quad_len], end_quad_input[-one_side_quad_len:]], axis=0)
  quad_input = np.concatenate([start_quad_input, end_quad_input], axis=0, dtype=np.int32)
  return quad_input

def create_features(idx_map,
                    input_ids,
                    code_vector_input_ids,
                    tokenizer,
                    seq_len=512,
                    disable_print=False):
  exists_input_ids = []
  exists_attention_masks = []

  code_input_ids = []
  code_attention_masks = []
  code_split_token = tokenizer('<c>')['input_ids'][1]

  for example in tqdm(idx_map, desc='Assemble Exists Features', disable=disable_print):
    code_block = []
    md_idx, code_idxs, doc_count = example[0], example[1], example[2]
    markdown_input_id = list(input_ids[md_idx])
    for code_idx in code_idxs:
      code_block += list(input_ids[code_idx]) + [code_split_token]
    code_block = code_block[:-1]

    exists_block = ([tokenizer.cls_token_id]
                    + markdown_input_id
                    + [tokenizer.sep_token_id]
                    + code_block
                    + [tokenizer.sep_token_id])
    code_vector = ([tokenizer.cls_token_id]
                   + markdown_input_id
                   + [tokenizer.sep_token_id]
                   + code_vector_input_ids[doc_count]
                   + [tokenizer.sep_token_id])

    # Exists Special Tokens
    attention = [1] * len(exists_block)
    no_attention = [0] * (seq_len - len(attention))
    input_id_padding = [tokenizer.pad_token_id] * (seq_len - len(exists_block))

    exists_block = exists_block + input_id_padding
    exists_attention_mask = attention + no_attention
    exists_input_ids.append(exists_block)
    exists_attention_masks.append(exists_attention_mask)

    # Code Special Tokens
    attention = [1] * len(code_vector)
    no_attention = [0] * (seq_len - len(attention))    
    input_id_padding = [tokenizer.pad_token_id] * (seq_len - len(code_vector))
    
    code_vector = code_vector + input_id_padding
    code_attention_mask = attention + no_attention
    code_input_ids.append(code_vector)
    code_attention_masks.append(code_attention_mask)

  exists_input_ids = np.asarray(exists_input_ids, dtype=np.int32)
  exists_attention_masks = np.asarray(exists_attention_masks, dtype=np.int32)
  code_input_ids = np.asarray(code_input_ids, dtype=np.int32)
  code_attention_masks = np.asarray(code_attention_masks, dtype=np.int32)     
  return [exists_input_ids, exists_attention_masks, code_input_ids, code_attention_masks]

def balance_shuffle_dataset(features, labels):
  rng = np.random.default_rng()
  
  pos_idx = np.where(labels == 1)[0]
  rng.shuffle(pos_idx)
  pos_exists_ii = np.take(features[0], pos_idx, axis=0)
  pos_exists_am = np.take(features[1], pos_idx, axis=0)
  pos_code_ii = np.take(features[2], pos_idx, axis=0)
  pos_code_am = np.take(features[3], pos_idx, axis=0)
  pos_labels = np.take(labels, pos_idx, axis=0)

  neg_class_size = int(len(pos_idx)*1.1)  # Introduce noise to the dataset
  neg_idx = np.where(labels == 0)[0]
  rng.shuffle(neg_idx)
  neg_exists_ii = np.take(features[0], neg_idx, axis=0)[:neg_class_size]
  neg_exists_am = np.take(features[1], neg_idx, axis=0)[:neg_class_size]
  neg_code_ii = np.take(features[2], neg_idx, axis=0)[:neg_class_size]
  neg_code_am = np.take(features[3], neg_idx, axis=0)[:neg_class_size]
  neg_labels = np.take(labels, neg_idx, axis=0)[:neg_class_size]

  features[0] = np.concatenate([pos_exists_ii, neg_exists_ii], axis=0)
  features[1] = np.concatenate([pos_exists_am, neg_exists_am], axis=0)
  features[2] = np.concatenate([pos_code_ii, neg_code_ii], axis=0)
  features[3] = np.concatenate([pos_code_am, neg_code_am], axis=0)
  labels = np.concatenate([pos_labels, neg_labels], axis=0)

  shuffler = np.arange(len(labels))
  rng.shuffle(shuffler)
  features[0] = np.take(features[0], shuffler, axis=0)
  features[1] = np.take(features[1], shuffler, axis=0)
  features[2] = np.take(features[2], shuffler, axis=0)
  features[3] = np.take(features[3], shuffler, axis=0)
  labels = np.take(labels, shuffler, axis=0)
  labels = np.reshape(labels, (len(labels), 1))
  return tuple(features), labels

def create_idx_map(cell_metadata, disable_print=False): 
    idx_map = []
    code_vectors = []
    doc_start = 0

    for doc_count, metadata in enumerate(tqdm(cell_metadata, desc='Creating Index Mapping', disable=disable_print)):
      code_count = list(metadata.values()).count('code')
      quarter_loc, remainder = divmod(code_count, 4)
      quadrant_splits = create_quadrant_splits(code_count)
      doc_code_vector = create_doc_code_vector(quarter_loc, quadrant_splits, doc_start)

      md_idxs = [idx
                 for idx, cell_id in enumerate(metadata)
                 if metadata[cell_id] == 'markdown']

      for quad_num in range(4):
        quad_range = np.arange(quadrant_splits[quad_num], quadrant_splits[quad_num+1]) + doc_start

        if len(quad_range) >= 9:
          quad_maps = [[md_idx + doc_start, list(quad_range[:4]) + list(quad_range[-3:]), doc_count]
                        for md_idx in md_idxs]
        else:
          quad_maps = [[md_idx + doc_start, list(quad_range), doc_count]
                        for md_idx in md_idxs]

        for quad_map in quad_maps:
          idx_map.append(quad_map)
      
      code_vectors.append(doc_code_vector)
      doc_start += len(metadata)
    return idx_map, code_vectors