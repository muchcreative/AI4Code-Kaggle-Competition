"""Creates and initalizes tensorflow models."""

import re
import math
import keras.layers

import tensorflow as tf
from tensorflow.keras import Model, Input, layers

class MDCodeRep(Model):
  def __init__(self, md_transformer, md_trans_name, quad_transformer, quad_trans_name):
    super(MDCodeRep, self).__init__(name='MDCodeRep')
    self.md_trans = TransformerBlock(md_transformer, 'md_'+md_trans_name)
    self.quad_trans = TransformerBlock(quad_transformer, 'quad_'+quad_trans_name)
    self.concat = layers.Concatenate(axis=-1)
    self.mlp = MLPBlock()
    self.pred_head = layers.Dense(1, activation='linear')  # consider relu activation

  def call(self, input_tensor):
    md_trans_output = self.md_trans(input_tensor[0], input_tensor[1])
    quad_trans_output = self.quad_trans(input_tensor[2], input_tensor[3])
    trans_output = self.concat([md_trans_output, quad_trans_output])
    trans_output = self.mlp(trans_output)
    prediction = self.pred_head(trans_output)
    return prediction

  def build_graph(self):
    x0 = Input(shape=(512,), dtype=tf.int32)
    x1 = Input(shape=(512,), dtype=tf.int32)
    x2 = Input(shape=(512,), dtype=tf.int32)
    x3 = Input(shape=(512,), dtype=tf.int32)
    x = tuple([x0, x1, x2, x3])
    return Model(inputs=x, outputs=self.call(x))

class MDOrder(Model):
  def __init__(self, transformer, model_name, trans_model):
    super(MDOrder, self).__init__(name=model_name)
    self.trans = TransformerBlock(transformer, trans_model)
    self.pred_head = layers.Dense(1, activation='sigmoid')

  def call(self, input_tensor):
    trans_output = self.trans(input_tensor[0], input_tensor[1])
    prediction = self.pred_head(trans_output)
    return prediction

  def build_graph(self):
    x0 = Input(shape=(512,), dtype=tf.int32)
    x1 = Input(shape=(512,), dtype=tf.int32)
    x = [x0, x1]
    return Model(inputs=x, outputs=self.call(x))

class MDQuad(Model):
  def __init__(self, transformer, model_name, trans_model):
    super(MDQuad, self).__init__(name=model_name)
    self.trans = TransformerBlock(transformer, trans_model)
    self.pred_head = layers.Dense(4, activation='softmax')

  def call(self, input_tensor):
    trans_output = self.trans(input_tensor[0], input_tensor[1])
    prediction = self.pred_head(trans_output)
    return prediction

  def build_graph(self):
    x0 = Input(shape=(512,), dtype=tf.int32)
    x1 = Input(shape=(512,), dtype=tf.int32)
    x = tuple([x0, x1])
    return Model(inputs=x, outputs=self.call(x))

class TransformerBlock(layers.Layer):
  def __init__(self, transformer, trans_model):
    super(TransformerBlock, self).__init__(name=trans_model)
    self.transformer = transformer
    first_token = ['bert', 'bert_ml', 'cbert', 't5']
    last_token = ['bart']
    
    if trans_model in first_token:
      self.token_loc = 0
    elif trans_model in last_token:
      self.token_loc = -1

  def call(self, input_ids, attention_mask):
    x = self.transformer(input_ids, attention_mask)[0][:,self.token_loc,:]
    return x

class MLPBlock(layers.Layer):
  def __init__(self):
    super(MLPBlock, self).__init__()
    self.dense1 = layers.Dense(768, activation='relu')
    self.dense2 = layers.Dense(192, activation='relu')

  def call(self, input_tensor):
    x = self.dense1(input_tensor)
    x = self.dense2(x)
    return x

class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, base_learning_rate, warmup_steps, total_steps):
    self.base_learning_rate = base_learning_rate
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps

  def __call__(self, step):
    return  (self.base_learning_rate
             * tf.cond(tf.math.less_equal(step, self.warmup_steps),
                       lambda: step / self.warmup_steps,
                       lambda: (step - self.total_steps) / (self.warmup_steps - self.total_steps)))
  
  def get_config(self):
    config = {'base_learning_rate': self.base_learning_rate,
              'warmup_steps': self.warmup_steps,
              'total_steps': self.total_steps}
    return config

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, base_learning_rate, warmup_steps, total_steps):
    self.base_learning_rate = base_learning_rate
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.decay_steps = total_steps - warmup_steps
    
  def __call__(self, step):
    return tf.cond(tf.math.less_equal(step, self.warmup_steps),
                    lambda: tf.multiply(self.base_learning_rate, step / self.warmup_steps),
                    lambda: tf.multiply(self.base_learning_rate, self.decayed_learning_rate(step)))

  def decayed_learning_rate(self, step):
    step = tf.minimum(step, self.decay_steps)
    decay_progress = step / self.decay_steps
    cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * decay_progress))
    return cosine_decay
  
  def get_config(self):
    config = {'base_learning_rate': self.base_learning_rate,
              'warmup_steps': self.warmup_steps,
              'total_steps': self.total_steps}
    return config

class NeptuneCallback(tf.keras.callbacks.Callback):
  def __init__(self, run):
    super().__init__()
    self.run = run

  def _log_metrics(self, logs, category, trigger):
    if not logs:
      return
    logger = self.run[category][trigger]
    for metric, value in logs.items():
      try:
        if metric in ('batch', 'size') or metric.startswith('val_'):
          continue
        logger[metric].log(value)
      except Exception as e:
        pass
  
  def on_train_batch_end(self, batch, logs=None):
    self._log_metrics(logs, 'train', 'batch')

  def on_epoch_end(self, epoch, logs=None):
    self._log_metrics(logs, 'train', 'epoch')

  def on_test_batch_end(self, batch, logs=None):
    self._log_metrics(logs, 'test', 'batch')

  def on_test_end(self, logs=None):
    self._log_metrics(logs, 'test', 'epoch')

def reinit_weights_and_bias(transformer, n_layers, layer_name='roberta'):
  """All layers get reiniatlized randomly based on their mean and config standard deviation
  LayerNorms get re-inailized to 1.0 and bias get re-initialized get 0.0
  
  Ensure all layers are trainable
  for i in range(11):
    print(transformer.get_layer('roberta').encoder.layer[i].trainable)
  """

  mean = 0.0
  std = transformer.config.initializer_range
  rng = np.random.default_rng()

  for n in range(n_layers):
    n_layer = transformer.get_layer(layer_name).encoder.layer[-(n+1)]
    layer_weights = reset_wb(n_layer, mean, std, rng)
    transformer.get_layer(layer_name).encoder.layer[-(n+1)].set_weights(layer_weights)
  return transformer

def reset_wb(n_layer, mean, std, rng):
  submodules = n_layer.submodules
  layer_weights = n_layer.weights

  reset_weights = []
  for i, submodule in enumerate(submodules):
    weight = layer_weights[i].numpy()

    if bool(re.search("bias", layer_weights[i].name)):
      weight.fill(0.0)
    elif isinstance(submodule, keras.layers.normalization.layer_normalization.LayerNormalization):
      weight.fill(1.0)
    else:
      weight = rng.normal(mean, std, weight.shape)
    reset_weights.append(weight)
  return reset_weights