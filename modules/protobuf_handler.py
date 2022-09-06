"""Handles ProtoBuf encoding and decoding."""

import re
from tqdm.auto import tqdm
import tensorflow as tf

### Encode ProtoBufs ###
def write_dataset_to_tfrecords(features, labels, filepath):
  """Parses features and labels and saves as a TFRecord file."""
  writer = tf.io.TFRecordWriter(filepath)

  for idx in tqdm(range(len(labels)), desc='Writing TFRecords to Disk'):
    current_protobuf = parse_protobuf(features[0][idx], 
                                      features[1][idx],
                                      labels[idx])
    writer.write(current_protobuf.SerializeToString())
  writer.close()
  print('\nAll examples have been written as tfrecords to disk')
  return

def parse_protobuf(input_ids, attention_mask, label):
  data = {'input_ids': _bytes_feature(serialize_tensor(input_ids)),
          'attention_mask': _bytes_feature(serialize_tensor(attention_mask)),
          'label': _bytes_feature(serialize_tensor(label))}
          
  protobuf = tf.train.Example(features=tf.train.Features(feature=data))
  return protobuf

def write_triple_set_to_tfrecords(features, labels, filepath):
  """Parses features and labels and saves as a TFRecord file."""
  writer = tf.io.TFRecordWriter(filepath)

  for idx in tqdm(range(len(labels)), desc='Writing TFRecords to Disk'):
    current_protobuf = parse_triple_protobuf(features[0][idx], 
                                             features[1][idx],
                                             features[2][idx],
                                             features[3][idx],
                                             features[4][idx],
                                             features[5][idx],
                                             labels[idx])
    writer.write(current_protobuf.SerializeToString())
  writer.close()
  print('\nAll examples have been written as tfrecords to disk')
  return

def write_dual_set_to_tfrecords(features, labels, filepath):
  """Parses features and labels and saves as a TFRecord file."""
  writer = tf.io.TFRecordWriter(filepath)

  for idx in tqdm(range(len(labels)), desc='Writing TFRecords to Disk'):
    current_protobuf = parse_dual_protobuf(features[0][idx], 
                                           features[1][idx],
                                           features[2][idx],
                                           features[3][idx],
                                           labels[idx])
    writer.write(current_protobuf.SerializeToString())
  writer.close()
  print('\nAll examples have been written as tfrecords to disk')
  return

def parse_triple_protobuf(c1_input_ids,
                          c1_attention_mask,
                          c2_input_ids,
                          c2_attention_mask,
                          c3_input_ids,
                          c3_attention_mask,
                          label):
  data = {'c1_input_ids': _bytes_feature(serialize_tensor(c1_input_ids)),
          'c1_attention_mask': _bytes_feature(serialize_tensor(c1_attention_mask)),
          'c2_input_ids': _bytes_feature(serialize_tensor(c2_input_ids)),
          'c2_attention_mask': _bytes_feature(serialize_tensor(c2_attention_mask)),
          'c3_input_ids': _bytes_feature(serialize_tensor(c3_input_ids)),
          'c3_attention_mask': _bytes_feature(serialize_tensor(c3_attention_mask)),
          'label': _bytes_feature(serialize_tensor(label))}
          
  protobuf = tf.train.Example(features=tf.train.Features(feature=data))
  return protobuf

def parse_dual_protobuf(c1_input_ids,
                        c1_attention_mask,
                        c2_input_ids,
                        c2_attention_mask,
                        label):
  data = {'c1_input_ids': _bytes_feature(serialize_tensor(c1_input_ids)),
          'c1_attention_mask': _bytes_feature(serialize_tensor(c1_attention_mask)),
          'c2_input_ids': _bytes_feature(serialize_tensor(c2_input_ids)),
          'c2_attention_mask': _bytes_feature(serialize_tensor(c2_attention_mask)),
          'label': _bytes_feature(serialize_tensor(label))}
          
  protobuf = tf.train.Example(features=tf.train.Features(feature=data))
  return protobuf

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_tensor(tensor):
  tensor = tf.io.serialize_tensor(tensor)
  return tensor

### Decode Protobufs ###
def sort_tfrecord_files(tfrecord_files):
  re_filter = '\d+'
  file_idx_start = [int(re.findall(re_filter, tfrecord_file.stem)[0]) for tfrecord_file in tfrecord_files]
  zipped_order = sorted(zip(file_idx_start, tfrecord_files))
  tfrecord_files = [tfrecord_file for _, tfrecord_file in zipped_order]
  return tfrecord_files

def decode_triple_protobuf(protobuf):
  schema = {'c1_input_ids': tf.io.FixedLenFeature([], dtype=tf.string),
            'c1_attention_mask': tf.io.FixedLenFeature([], dtype=tf.string),
            'c2_input_ids': tf.io.FixedLenFeature([], dtype=tf.string),
            'c2_attention_mask': tf.io.FixedLenFeature([], dtype=tf.string),
            'c3_input_ids': tf.io.FixedLenFeature([], dtype=tf.string),
            'c3_attention_mask': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.string)}
  return tf.io.parse_single_example(protobuf, schema)

def decode_dual_protobuf(protobuf):
  schema = {'c1_input_ids': tf.io.FixedLenFeature([], dtype=tf.string),
            'c1_attention_mask': tf.io.FixedLenFeature([], dtype=tf.string),
            'c2_input_ids': tf.io.FixedLenFeature([], dtype=tf.string),
            'c2_attention_mask': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.string)}
  return tf.io.parse_single_example(protobuf, schema)

def parse_triple_tensor_arrays(encoded_tensors):
  """Parse ensemble"""
  input_ids = [tf.reshape(tf.io.parse_tensor(encoded_tensors['c1_input_ids'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c1_attention_mask'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c2_input_ids'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c2_attention_mask'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c3_input_ids'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c3_attention_mask'], out_type=tf.int32), shape=(512,))]
  example_label = tf.reshape(tf.io.parse_tensor(encoded_tensors['label'], out_type=tf.float32), shape=(1,))
  return tuple(input_ids), example_label

def parse_dual_tensor_arrays(encoded_tensors):
  """Parse ensemble"""
  input_ids = [tf.reshape(tf.io.parse_tensor(encoded_tensors['c1_input_ids'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c1_attention_mask'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c2_input_ids'], out_type=tf.int32), shape=(512,)),
               tf.reshape(tf.io.parse_tensor(encoded_tensors['c2_attention_mask'], out_type=tf.int32), shape=(512,))]
  example_label = tf.reshape(tf.io.parse_tensor(encoded_tensors['label'], out_type=tf.float32), shape=(1,))
  return tuple(input_ids), example_label