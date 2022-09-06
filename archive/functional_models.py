'''Creates and initalizes tensorflow models.'''

import tensorflow as tf
from tensorflow.keras import Model, Input, layers

def build_pairwise_model(model_name, model_class, config_class):
    config = config_class(dropout=0.2, attention_dropout=0.2)
    code_transformer = model_class.from_pretrained(model_name, config=config)
    markdown_transformer = model_class.from_pretrained(model_name, config=config)
    
    x1 = Input(shape=(50,), dtype='int64', name="c_input_ids")
    x2 = Input(shape=(50,), dtype='int64', name="c_attention_mask")
    c_output = code_transformer([x1, x2])[0][:,0,:]  # Index [0] for the last hidden state, then index [:,0,:] for the [CLS] token
    c_output = layers.Dense(384, activation='relu', name='c_dense1')(c_output)
    c_output = layers.Dense(192, activation='relu', name='c_dense2')(c_output)

    x3 = Input(shape=(50,), dtype='int64', name="md_input_ids")
    x4 = Input(shape=(50,), dtype='int64', name="md_attention_mask")
    md_output = markdown_transformer([x3, x4])[0][:,0,:]
    md_output = layers.Dense(384, activation='relu', name='md_dense1')(md_output)
    md_output = layers.Dense(192, activation='relu', name='md_dense2')(md_output)
    
    c_md_output = layers.Concatenate(name='c_md_concatenation')([c_output, md_output])
    c_md_output = layers.Dense(192, activation='sigmoid', name='c_md_dense1')(c_md_output)
    c_md_output = layers.Dense(96, activation='sigmoid', name='c_md_dense2')(c_md_output)
    c_md_output = layers.Dense(12, activation='sigmoid', name='c_md_dense3')(c_md_output)
    predictions = layers.Dense(1, activation='sigmoid', name='pred_head')(c_md_output)

    model = Model(inputs=(x1, x2, x3, x4),
                  outputs=predictions,
                  name='Pairwise_Dual_Bert_Model')
    return model

def build_twin_pointwise_model(model_name, model_class, config_class):
    config = config_class(dropout=0.2, attention_dropout=0.2)
    code_transformer = model_class.from_pretrained(model_name, config=config)
    markdown_transformer = model_class.from_pretrained(model_name, config=config)
    
    x1 = Input(shape=(50,), dtype='int64', name="c_input_ids")
    x2 = Input(shape=(50,), dtype='int64', name="c_attention_mask")

    c_output = code_transformer([x1, x2])[0][:,0,:]  # (input_size, 768) hidden state / output shape
    c_output = layers.Dense(384, activation='relu', name='c_dense1')(c_output)
    c_output = layers.Dense(192, activation='relu', name='c_dense2')(c_output)
    c_output = layers.Dense(96, activation='relu', name='c_dense3')(c_output)
    c_output = layers.Dense(48, activation='relu', name='c_dense4')(c_output)
    c_output = layers.Dense(24, activation='relu', name='c_dense5')(c_output)

    x3 = Input(shape=(50,), dtype='int64', name="md_input_ids")
    x4 = Input(shape=(50,), dtype='int64', name="md_attention_mask")
    md_output = markdown_transformer([x3, x4])[0][:,0,:]
    md_output = layers.Dense(384, activation='relu', name='md_dense1')(md_output)
    md_output = layers.Dense(192, activation='relu', name='md_dense2')(md_output)
    md_output = layers.Dense(96, activation='relu', name='md_dense3')(md_output)
    md_output = layers.Dense(48, activation='relu', name='md_dense4')(md_output)
    md_output = layers.Dense(24, activation='relu', name='md_dense5')(md_output)
    
    c_md_output = layers.Concatenate(name='c_md_concatenation')([c_output, md_output])
    c_md_output = layers.Dense(12, activation='sigmoid', name='c_md_dense1')(c_md_output)
    predictions = layers.Dense(1, activation='sigmoid', name='pred_head')(c_md_output)

    model = Model(inputs=(x1, x2, x3, x4),
                  outputs=predictions,
                  name='Twin_Pointwise_Bert_Model')
    return model

class InBetweenLTR(Model):
  """In-between LTR Model"""
  def __init__(self, model_name, model_class, config_class):
    super(InBetweenLTR, self).__init__()
    self.bart = BartBlock(model_name, model_class, config_class)
    self.mlp = MLPBlock()

  def call(self, input_tensor):
    bart_output = self.bart(input_tensor)
    prediction = self.mlp(bart_output)
    return prediction

  def build_graph(self):
    x1 = Input(shape=(150,), dtype=tf.int32)  # There are 3 main inputs altogther with input_ids and attention_masks each
    x2 = Input(shape=(150,), dtype=tf.int32)
    x3 = Input(shape=(150,), dtype=tf.int32)
    x4 = Input(shape=(150,), dtype=tf.int32)
    x5 = Input(shape=(150,), dtype=tf.int32)
    x6 = Input(shape=(150,), dtype=tf.int32)
    x = [x1, x2, x3, x4, x5, x6]
    return Model(inputs=x, outputs=self.call(x))

class BartBlock(layers.Layer):
  def __init__(self, model_name, model_class, config_class):
    super(BartBlock, self).__init__()
    self.concat = layers.Concatenate(axis=1)
    self.config = config_class()
    self.transformer = model_class.from_pretrained(model_name)  # Config has issues right now

  def call(self, input_tensor):
    input_ids = [input_tensor[0], input_tensor[2], input_tensor[4]]
    attention_mask = [input_tensor[1], input_tensor[3], input_tensor[5]]
    return self.transformer(input_ids, attention_mask)[0][:,-1,:]  # Last output on BART contains the context and understanding of the sequences

class MLPBlock(layers.Layer):
  def __init__(self):
    super(MLPBlock, self).__init__()
    self.dense1 = layers.Dense(192, activation='relu')
    self.dense2 = layers.Dense(96, activation='relu')
    self.dense3 = layers.Dense(3, activation='softmax')

  def call(self, input_tensor):
    x = self.dense1(input_tensor)
    x = self.dense2(x)
    x = self.dense3(x)
    return x

