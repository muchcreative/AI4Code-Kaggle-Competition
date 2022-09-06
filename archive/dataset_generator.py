# Intailize data pipeline / generator
# Try less documents and only uploading equal to the batch size

class DataGenerator:
    '''Loads data from json files.
    
    No way you never use a py.function huh? 
        
    Yields:
        dataset: generated tensorflow dataset
        dataset given as (features, labels), can we avoid a tf.data.Dataset.zip()?
    '''
            
    def __iter__(self):
        for start_idx in range(0, total_data_points, data_per_yield):
            end_idx = start_idx + data_per_yield
            doc_ids, cell_metadata, text = self.load_and_parse_data(data_paths, start_idx, end_idx)
            input_ids, attention_mask = self.encode_text(text, tokenizer)
        
            idx_map = self.create_idx_map(cell_metadata)
            features = self.create_features(input_ids, attention_mask, idx_map)
            labels = self.create_labels(doc_ids,
                                        cell_metadata,
                                        df_orders,
                                        idx_map)
            yield features, labels

    def load_and_parse_data(self, data_paths, start_idx, end_idx):
        '''Loads and parses the json files at the given data_paths.
    
        Returns:
            doc_ids: list of strings containing the unique document ID.
            cell_metadata: list of dictionaries containing the {cell_id: cell_type}
                        for each document
            text: list of lists containing the entire unordered text data for each document
                  for each list. Each inner list element contains either a code cell or markdown text.
        '''
    
        generated_data_paths = data_paths[start_idx:end_idx]
    
        doc_ids = []
        cell_metadata = []
        text = []
    
        for data_path in tqdm(generated_data_paths, desc='Loading Json Files'):
            data = json.loads(data_path.read_bytes())  # Opens the file, read the bytes, and then closes the file
            text += list(data['source'].values())
        
            doc_ids.append(os.path.basename(data_path).replace(".json",""))
            cell_metadata.append(data['cell_type'])
        return doc_ids, cell_metadata, text
        
    def encode_text(self, text, tokenizer):
        '''Encodes text for the transformer model using the specified tokenizer.'''
        encoded_text = tokenizer(text,
                                 max_length=200, # Consider setting to one deviation higher than median or mean
                                 padding=True,
                                 truncation=True,
                                 return_tensors='tf')
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        return input_ids, attention_mask
    
    def create_idx_map(self, cell_metadata):
        '''Create an Index Map.
    
        Mapping will be used to index the features and labels
        First column will be the code cells and the second column will be the markdown cells
        Groupings can be shared with ancestors cells as well, they will get labelled the same value, can use a dict for this
    
        Returns:
            index_map: 2-D array, first column with code cells and second column for markdown cells
        '''
    
        idx_map = []
        doc_start = 0
    
        for metadata in tqdm(cell_metadata, desc='Creating Index Mapping'):
            code_idxs = [idx
                         for idx, cell_id in enumerate(metadata)
                         if metadata[cell_id] == 'code']
            markdown_idxs = [idx
                             for idx, cell_id in enumerate(metadata)
                             if metadata[cell_id] == 'markdown']       
            doc_map = [[doc_start+code_idx, doc_start+markdown_idx]
                        for code_idx in code_idxs
                        for markdown_idx in markdown_idxs]
            doc_start += len(metadata)
            idx_map.append(np.asarray(doc_map))  # Setting as np arrays helps with slicing and indexing later
        
        idx_map = tf.concat(idx_map, axis=0)  # Flatten list of numpy arrays to a 2-D tensor
        tf.random.set_seed(seed=24)  # Set global random seed
        idx_map = tf.random.shuffle(idx_map, seed=42)[:100000]  # Take only first 100000 elements from the shuffle for uniform batch sizing
        return idx_map

    def create_features(self, input_ids, attention_mask, idx_map):
        code_input_ids = tf.gather(input_ids, indices=idx_map[:, 0], axis=0)
        code_attention_mask = tf.gather(attention_mask, indices=idx_map[:, 0], axis=0)
    
        markdown_input_ids = tf.gather(input_ids, indices=idx_map[:, 1], axis=0)
        markdown_attention_mask = tf.gather(attention_mask, indices=idx_map[:, 1], axis=0)
        
        features = (code_input_ids,
                    code_attention_mask,
                    markdown_input_ids,
                    markdown_attention_mask)
        return features

    def create_labels(self,
                      doc_ids,
                      cell_metadata,
                      df_orders,
                      idx_map):
        '''Creates labels by comparing the actual ordering with the index map.
    
        Args:
            doc_ids:
            cell_metadata:
            df_orders:
            idx_map:

        Returns:
            labels: 1-D binary array. If the code line appears first in the document compared
                    to the markdown line, the labels is 1, else 0.    
        '''
    
        doc_start = 0
        orders = []
    
        for counter, doc_id in enumerate(doc_ids):
            ordered_cells = df_orders.loc[doc_id]
            doc_order = np.asarray([ordered_cells.index(cell_id) for cell_id in cell_metadata[counter]])
            orders.append(doc_order)

        orders = tf.concat(orders, axis=0)
        labels = tf.where(tf.gather(orders, idx_map[:, 0])
                          < tf.gather(orders, idx_map[:, 1]),
                          1, 0)
        return labels