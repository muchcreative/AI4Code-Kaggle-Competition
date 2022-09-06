import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import os
import zipfile

# Check if notebook can be added to the dataset
def is_python_notebook(nb_file):
  try:
    nb_ver = nb_file['metadata']['language_info']['version']
    nb_lang = nb_file['metadata']['kernelspec']['name']
    version_3 = nb_ver.startswith('3.') 
    in_python = bool(re.search('python', nb_lang))
    return version_3 and in_python 
  except:
    return False

def has_min_cell_types(nb_file):
  md_count = 0
  code_count = 0
  try:
    for cell in nb_file['cells']:
      if cell['source'] == []:
        pass
      elif cell['cell_type'] == 'markdown':
        md_count += 1
      elif cell['cell_type'] == 'code':
        code_count += 1
  except:
    return False
  return (code_count >= 1 and md_count >= 1)

def is_ml_related(nb_file):
  ml_pattern = re.compile(r'sklearn|keras|tensorflow|pytorch|torch|seaborn|matplotlib|numpy|pandas|scipy|xgboost|lightgbm|fastai|kaggle')
  for cell in nb_file['cells']:
    try:
      if bool(re.search(ml_pattern, cell['source'][0])):
        return True
    except:
      pass
  return False

def useable_notebook(nb_file):
  if is_python_notebook(nb_file):
    if has_min_cell_types(nb_file):
      if is_ml_related(nb_file):
        return True
  return False

def setup_clean_notebook(current_doc, count):
  current_doc = current_doc.drop(columns=['metadata', 'outputs', 'execution_count'])
  current_doc = current_doc[(current_doc['cell_type'] == 'code') | (current_doc['cell_type'] == 'markdown')] 

  # Clean Text
  empty_text_pattern = re.compile('^[ \t\n]*$')
  current_doc["source"] = current_doc["source"].apply(convert_list_to_text)
  current_doc['source'] = current_doc['source'].replace(empty_text_pattern, np.nan)
  current_doc = current_doc.dropna()

  # Setup Ranks
  current_doc = current_doc.reset_index(drop=True)
  current_doc['rank'] = current_doc.index
  current_doc = current_doc.sort_values(by=['cell_type', 'rank'])

  # Check for minimum cell type counts
  value_counts = current_doc['cell_type'].value_counts()
  try:
    code_count = value_counts['code']
    md_count = value_counts['markdown']
  except:
    return []

  # Shuffle markdowns
  md_rows = current_doc[code_count:]
  md_rows = md_rows.sample(frac=1)
  current_doc[code_count:] = md_rows

  # Create cell_ids
  code_ids = create_ids(code_count, count, prefix='current_kag_code_id_')
  md_ids = create_ids(md_count, count, prefix='current_kag_md_id_')
  current_doc['cell_id'] = code_ids + md_ids

  if md_count >= 1 and code_count >= 1:
    return current_doc
  else:
    return []

def convert_list_to_text(text):
  if type(text) == str:
    return text
  else:
    text = " ".join(text)
  return text

def create_ids(length, count, prefix):
  return [prefix + str(i) for i in range(count, count+length)]

def temp_clean_df(current_df):
  """Clean the rest of the empty cells you missed."""
  initial_len = len(current_df)

  # Clean Text
  empty_text_pattern = re.compile('^[ \t\n]*$')
  current_df['source'] = current_df['source'].replace(empty_text_pattern, np.nan)
  current_df = current_df.dropna()

  # If nothing changed, just return the current_df
  if initial_len == len(current_df):
    return current_df

  # Re-Index dataframe
  current_df = current_df.sort_values("rank")
  current_df = current_df.reset_index(drop=True)
  current_df['rank'] = current_df.index
  current_df = current_df.sort_values(by=['cell_type', 'rank'])

  # Check for minimum cell type counts
  value_counts = current_df['cell_type'].value_counts()
  try:
    code_count = value_counts['code']
    md_count = value_counts['markdown']
  except:
    return None

  # Shuffle markdowns
  md_rows = current_df[code_count:]
  md_rows = md_rows.sample(frac=1)
  current_df[code_count:] = md_rows

  if md_count >= 1 and code_count >= 1:
    return current_df
  else:
    return None

def zip_data_folder(zip_location, folder_path):
  """zip location must have '.zip' added to it"""
  with zipfile.ZipFile(zip_location, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
    for folder_name, subfolders, filenames in os.walk(folder_path):
      for filename in tqdm(filenames, desc='Zipping Files'):
        file_path = os.path.join(folder_name, filename)
        zip_ref.write(file_path, arcname=os.path.relpath(file_path, folder_path))
  print(f'Succesfully zipped files to {zip_location}')

# doc_num = 0
# save_path = '/content/jup_train'
# os.makedirs(save_path)

# for data_path in tqdm(juypter_paths, desc='Re-Cleaning Data'):
#   current_df = pd.read_csv(data_path)
#   current_df = data_cleaner.temp_clean_df(current_df)
#   if current_df is not None:
#     current_df.to_csv(save_path + f'/jup_data_{doc_num}.csv', index=False)
#     doc_num += 1

# zip_location = '/content/jup_train.zip'
# folder_path = save_path
# zip_data_folder(zip_location, folder_path)
# folder_path = save_path

# data_cleaner.zip_data_folder(zip_location, folder_path)