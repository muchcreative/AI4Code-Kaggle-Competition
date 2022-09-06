def preprocess_text(text, cell_metadata, verbose=1):
  """Preprocesses markdown and code text using different processors."""

  if verbose == 0:
    disable_print = True
  else:
    disable_print = False

  try:
    nltk.data.find('corpora/wordnet')
  except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
  stemmer = WordNetLemmatizer()
  
  flattened_code_types = [cell_type
                          for metadata in cell_metadata
                          for cell_type in metadata.values()]

  preprocessed_text = [preprocess_line(text[idx], stemmer)
                      if cell_type == 'markdown'
                      else text[idx]
                      for idx, cell_type in enumerate(tqdm(flattened_code_types, desc='Preprocessing Text', disable=disable_print))]
  return preprocessed_text

def preprocess_line(line, stemmer):
  '''Preprocesses a single line of text.'''
  
  # Remove XML text
  # re.sub('<[^<]+>', '', xml_text)  # Right now you still leave span and the like, you should process those out too

  # Remove all the special characters
  line = re.sub(r'\W', ' ', str(line))

  # Remove all single characters
  line = re.sub(r'\s+[a-zA-Z]\s+', ' ', line)

  # Remove single characters from the start
  line = re.sub(r'\^[a-zA-Z]\s+', ' ', line)

  # Substituting multiple spaces with single space
  line = re.sub(r'\s+', ' ', line, flags=re.I)

  # Removing prefixed 'b'
  line = re.sub(r'^b\s+', '', line)

  # Converting to Lowercase
  line = line.lower()
  
  # Lemmatization
  tokens = line.split()
  tokens = [stemmer.lemmatize(word) for word in tokens]
  tokens = [word for word in tokens if len(word) > 2]

  preprocessed_line = ' '.join(tokens)
  return preprocessed_line