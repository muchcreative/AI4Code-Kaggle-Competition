'''Converts Data to TFRecords Format.'''

import math
import numpy as np
import pandas as pd
import tensorflow as tf

import os
from pathlib import Path
import zipfile

import json
import yaml
from tqdm.auto import tqdm

import re
import unicodedata

def check_for_tpu_status():
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  tpu_strategy = tf.distribute.TPUStrategy(resolver)
  print("TPU Strategy has been initialized and setup")
  return tpu_strategy

def check_for_gpu_status():
  '''Checks if you are using a GPU runtime.'''
  device_name = tf.test.gpu_device_name()  # !nvidia-smi shell command can check GPU type
  if device_name != '/device:GPU:0':
    print('GPU device not found.')
  else:
    print(f'Found GPU at: {device_name}.')

def unzip_files(data_path, disk_path):
  if Path(disk_path).exists():
    print('Files have already been unzipped to disk path.')
  else:
    os.makedirs(disk_path)
    zip_ref = zipfile.ZipFile(data_path)
    for member in tqdm(zip_ref.infolist(), desc='Unzipping files'):
      zip_ref.extract(member, path=disk_path)
    print('\n Done unzipping data to disk path.')
  return 

def load_yaml_file(yaml_path):
  with open(yaml_path, 'r') as stream:
    try:
      yaml_file = yaml.safe_load(stream)
      print('Safe loaded in yaml file.')
    except yaml.YAMLError as error:
      print(error)
  return yaml_file

def remove_id_paths(data_paths, excluded_ids):
  filtered_data_paths = [data_path 
                        for data_path in tqdm(data_paths, desc='Removing Specified Ids')
                        if (data_path.stem not in excluded_ids)]
  return filtered_data_paths

def load_and_parse_data(data_paths, start_idx, end_idx):
    generated_data_paths = data_paths[start_idx:end_idx]
    
    doc_ids = []
    cell_metadata = []
    text = []
    
    for data_path in tqdm(generated_data_paths, desc='Loading Json Files'):
        current_id = data_path.stem
        if bool(re.search('kag|jup', current_id)):
          data = pd.read_csv(data_path, index_col='cell_id')
          doc_ids.append(current_id)
          cell_metadata.append(data['cell_type'].to_dict())
          text += list(data['source'])
        else:
          data = json.loads(data_path.read_bytes())  # Opens the file, read the bytes, and then closes the file
          doc_ids.append(current_id)
          cell_metadata.append(data['cell_type'])
          text += list(data['source'].values())
    return doc_ids, cell_metadata, text

def add_custom_tokens_to_tokenizer(tokenizer, custom_tokens):
  tokenizer.add_tokens(custom_tokens)
  print('Custom tokens have been added to the tokenizer')
  return tokenizer

def preprocess_text(text, code_markdown_locs, disable_print=False):
  preprocessed_text = [markdown_preprocessing(text[idx])
                       if cell_type == 'markdown'
                       else code_preprocessing(text[idx])
                       for idx, cell_type in enumerate(tqdm(code_markdown_locs, desc='Preprocessing Text', disable=disable_print))]
  return preprocessed_text

def code_preprocessing(line):
  if line is np.nan:
    return '[EMPTY]'
  line = re.sub(r'\\n', '\n', line)
  line = re.sub(r'\n\n', '\n', line)
  line = re.sub(r'^[ \t\n]*$', '[EMPTY]', line)
  line = re.sub(r' +', ' ', line)
  return line

def markdown_preprocessing(line):
  if line is np.nan:
    return '[EMPTY]'
  # Remove markup links like [name](link)
  line = re.sub(r'\]\([^)]*\)', '](', line)
  line = re.sub(r'\[([^\]\(]*)\]\(', r'\1', line)

  # And the commonly mistaken version too lol
  line = re.sub(r'\)\[[^]]*\]', ')[', line)
  line = re.sub(r'\(([^\)\[]*)\)\[', r'\1', line)
  
  # keep img, a, h, change to comma points. Choose
  # What you want to keep first,
  line = re.sub(r'<div .+?(?=[id|name])[id|name] *= *["\']([^"\']+)["\']*[^>]*>', r' Markup with name \1. ', line)
  line = re.sub(r'<a .+?(?=[id|name])[id|name] *= *["\']([^"\']+)["\']*[^>]*>', r' Markup with name \1. ', line)
  line = re.sub(r'<img .+?(?=alt)[alt]* *= *["\']([^"\']+)["\']*[^>]*>', r' Image Markup with name \1. ',line)
  line = re.sub(r'<img[^<>]+>', ' Image Markup. ', line)

  # Clean xml language formatting
  # b, div, span, ul, p, table, td, tr, th
  line = re.sub(r'<hr>', '[DIVIDER]', line)
  line = re.sub(r'^<br>', '[DIVIDER]', line)
  line = re.sub(r'<[^<>]+>', ' ', line)
  line = re.sub(r'&nbsp;', ' ', line)

  # Remove any hashtags
  line = re.sub(r'#*', '', line)

  # Changes common dividers to the <divider> token
  line = re.sub(r'---+|\*\*\*+|___+', ' [DIVIDER] ', line)
  
  line = re.sub(r'\\n>', '[DIVIDER]', line)
  line = re.sub(r'^>', '[DIVIDER]', line)

  # Remove '*' that allow bolding and italices
  line = re.sub(r'\**', r'', line)

  # Remove emojis, but Bart and RoBERTa can understand emojis
  emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U0001F914"             # Thinking face emoji  
                              "]+", flags=re.UNICODE)
  line = emoji_pattern.sub(r' [EMOJI] ', line)
  line = re.sub(r':P|:D|:\)|: \)', ' [EMOJI] ', line)  # Change :P, :D, :), or : ), change to <emoji>  
 
  # Change circled numbers like ① to 1
  line = unicodedata.normalize('NFKC', line)

  # Remove additional spaces and indents
  line = re.sub(r'\s+', ' ', line, flags=re.I)

  # Removes the inital and last spaces and indents
  line = re.sub(r'^[ \s]+|[ \s]+$', '', line)
  
  # Remove common special characters -, ., _, :, ',' , `, ", ', ), (, 。
  if len(line) < 4:
     line = re.sub(r'\.*|,*|_*|-*|:*|`*|"*|\'*|\(*|\)*|。*', '', line)
     line = re.sub(r'^\!+|^\?+', '[EMOJI]', line)

  if len(line) == 0:
    line = '[EMPTY]'
  
  line = re.sub(r'^[ \t\n]*$', '[EMPTY]', line)
  return line

def encode_text_for_input_ids(text, tokenizer, disable_print=False):
  """Encodes text for the transformer model using the specified tokenizer."""
  input_ids = [tokenizer(line,
                         max_length=None,
                         padding=False,
                         truncation=True,
                         add_special_tokens=False,
                         return_attention_mask=False)['input_ids']
              for line in tqdm(text, desc='Encoding Text for Input Ids', disable=disable_print)]
  return input_ids

def get_orders(doc_ids, cell_metadata, df_orders):
  orders = []

  for counter, doc_id in enumerate(doc_ids):
    ordered_cell_ids = df_orders.loc[doc_id]
    ordered_cell_types = [cell_metadata[counter][cell_id] for cell_id in ordered_cell_ids]
    doc_cell_ids = list(cell_metadata[counter].keys())

    ranks = {}
    current_rank = -1  # First position is 0
    in_markdown_block = False

    for idx, cell_id in enumerate(ordered_cell_ids):
      if ordered_cell_types[idx] == 'code':
        current_rank += 1
        ranks[cell_id] = int(current_rank)
        in_markdown_block = False

      elif ordered_cell_types[idx] == 'markdown' and not in_markdown_block:
        current_rank += 1
        ranks[cell_id] = current_rank
        in_markdown_block = True
        
      elif ordered_cell_types[idx] == 'markdown' and in_markdown_block:
        ranks[cell_id] = current_rank
        markdown_block = True

    unordered_ranks = [ranks[cell_id] for cell_id in doc_cell_ids]
    orders.append(unordered_ranks)

  orders = np.concatenate(orders, axis=0)
  return orders

def create_quadrant_splits(code_cell_count):
  quarter_loc, remainder = divmod(code_cell_count, 4)
  quadrant_splits = np.arange(0, code_cell_count-remainder+0.1, step=quarter_loc, dtype=np.int32)
  for i in range(0, remainder):
    quadrant_splits[i+1:] = quadrant_splits[i+1:] + 1
  return quadrant_splits

def get_ordered_df(doc_id, df, df_orders):
  """Creates the ordered pandas dataframe with the ordered text."""
  return df.loc[doc_id].loc[df_orders[doc_id]]