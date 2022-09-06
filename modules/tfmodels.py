"""Creates and initalizes tensorflow models."""

import re
import math
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input, layers, backend

class TripleCodeRep(Model):
  def __init__(self, c1_transformer, c2_transformer, c3_transformer):
    super(TripleCodeRep, self).__init__(name='TripleCodeRep')
    self.c1_trans = TransformerBlock(c1_transformer, prefix='c1_')
    self.c2_trans = TransformerBlock(c2_transformer, prefix='c2_')
    self.c3_trans = TransformerBlock(c3_transformer, prefix='c3_')

    self.cls_concat = layers.Concatenate(name='cls_concat')
    self.doc_concat = layers.Concatenate(name='doc_concat')

    self.cls_process = TokenProcess(name='cls_process')
    self.doc_process = TokenProcess(name='doc_process')

    self.final_concat = layers.Concatenate(name='final_concat')
    self.pred_head = layers.Dense(1, activation='tanh', name='pred_head')

  def call(self, input_tensor):
    c1_cls, c1_doc = self.c1_trans(input_tensor[0], input_tensor[1])
    c2_cls, c2_doc = self.c2_trans(input_tensor[2], input_tensor[3])
    c3_cls, c3_doc = self.c3_trans(input_tensor[4], input_tensor[5])
    
    cls = self.cls_concat([c1_cls, c2_cls, c3_cls])
    doc = self.doc_concat([c1_doc, c2_doc, c3_doc])

    cls = self.cls_process(cls)
    doc = self.doc_process(doc)

    x = self.final_concat([cls, doc])
    x = self.pred_head(x)
    return x

  def build_graph(self):
    x1 = Input(shape=(512,), dtype=tf.int32)
    x2 = Input(shape=(512,), dtype=tf.int32)
    x3 = Input(shape=(512,), dtype=tf.int32)
    x4 = Input(shape=(512,), dtype=tf.int32)
    x5 = Input(shape=(512,), dtype=tf.int32)
    x6 = Input(shape=(512,), dtype=tf.int32)
    x = tuple([x1, x2, x3, x4, x5, x6])
    return Model(inputs=x, outputs=self.call(x))

class TransformerBlock(layers.Layer):
  def __init__(self, transformer, prefix):
    super(TransformerBlock, self).__init__(name=prefix+'transformer')
    self.transformer = transformer
    self.prefix = prefix

    if self.prefix in ['c1_', 'c2_']:
      self.doc_token_loc = 72
    elif self.prefix in ['c3_']:
      self.doc_token_loc = 132

  def call(self, input_ids, attention_mask):
    x = self.transformer(input_ids, attention_mask)[0]
    cls_token = x[:,0,:]
    doc_token = x[:,self.doc_token_loc,:]
    return cls_token, doc_token

class TokenProcess(layers.Layer):
  def __init__(self, name):
    super(TokenProcess, self).__init__(name=name)
    self.dense1 = layers.Dense(768, activation='swish', name=name+'_dense1')
    self.dropout = layers.Dropout(0.1, name=name+'_dropout')
    self.dense2 = layers.Dense(384, activation='swish', name=name+'_dense2')
    self.norm = layers.LayerNormalization(name=name+'_norm')

  def call(self, input_tensor):
    x = self.dense1(input_tensor)  
    x = self.dropout(x)
    x = self.dense2(x)
    x = self.norm(x)
    return x

class WarmupCosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, base_learning_rate, warmup_steps, steps_per_epoch):
    self.base_learning_rate = base_learning_rate
    self.warmup_steps = warmup_steps

    self.steps_per_epoch = steps_per_epoch
    self.steps_in_epoch_with_warmup = steps_per_epoch - warmup_steps
    
  def __call__(self, step):
    return tf.cond(tf.math.less_equal(step, self.steps_per_epoch),
                   lambda: self.cosine_restart_with_warmup(step),
                   lambda: self.cosine_restart(step))
    
  def cosine_restart_with_warmup(self, step):
    return tf.cond(tf.math.less_equal(step, self.warmup_steps),
                    lambda: tf.multiply(self.base_learning_rate, step / self.warmup_steps),
                    lambda: tf.multiply(self.base_learning_rate, self._decayed_learning_rate_with_warmup(step)))

  def cosine_restart(self, step):
    current_epoch = tf.math.floordiv(step, self.steps_per_epoch)
    previous_epoch_steps = tf.math.multiply(current_epoch, self.steps_per_epoch)
    current_epoch_steps = tf.math.subtract(step, previous_epoch_steps)
    return tf.multiply(self.base_learning_rate, self._decayed_learning_rate(current_epoch_steps))

  def _decayed_learning_rate_with_warmup(self, step):
    step = tf.math.subtract(step, self.warmup_steps)
    decay_progress = tf.math.divide(step, self.steps_in_epoch_with_warmup)
    cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * decay_progress))
    return cosine_decay

  def _decayed_learning_rate(self, step):
    decay_progress = tf.math.divide(step, self.steps_per_epoch)
    cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * decay_progress))
    return cosine_decay
  
  def get_config(self):
    config = {'base_learning_rate': self.base_learning_rate,
              'warmup_steps': self.warmup_steps,
              'step_per_epoch': self.steps_per_epoch}
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

  for n in range(n_layers):
    encoder_layer = transformer.get_layer(layer_name).encoder.layer[-(n+1)]
    reset_weights = reset_wb(encoder_layer, mean, std)
    transformer.get_layer(layer_name).encoder.layer[-(n+1)].set_weights(reset_weights)
  return transformer

def reset_wb(encoder_layer, mean, std):
  layer_weights = encoder_layer.weights
  reset_weights = []
    
  for weight in layer_weights:
    shape, dtype = weight.shape, weight.dtype
        
    if bool(re.search("bias", weight.name)):
      weight = tf.zeros(shape, dtype)
    elif bool(re.search("LayerNorm", weight.name)):
      weight = tf.ones(shape, dtype)
    else:
      weight = tf.random.normal(shape, mean, std, dtype)
    reset_weights.append(weight)
  return reset_weights

class LRTracker(tf.keras.metrics.Metric):
  def __init__(self, optimizer, name='LRTracker'):
    super(LRTracker, self).__init__(name=name)
    self.current_lr = self.add_weight(name='tp', initializer='zeros')
    self.optimizer = optimizer

  def update_state(self, y_true, y_pred, sample_weight=None):
   pass

  def result(self):
    return self.optimizer._decayed_lr(tf.float32)
  
  def get_config(self):
    config = {'optimizer': self.optimizer}
    return config