#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# the first half of this notebook trains and saves a model
# the second half tests a saved model.
# note: the first half saves a model with a different name then is used in the second half so the current saved model isn't overwritten.

import numpy as np
import json
from sklearn.model_selection import train_test_split

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features

from transformers.configuration_bert import BertConfig

INFILE_prefix = '../data/prelabeled/'

# In[ ]:
# These two optiosn can increase training time
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})


# In[ ]:
tf.config.experimental.list_physical_devices()

# In[ ]:
# Model Components
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig("../models/BERT-config.json")
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)

# Data sets
train_ds = tf.data.TFRecordDataset(INFILE_prefix + "train47_even.tfrecord")
val_ds = tf.data.TFRecordDataset(INFILE_prefix + "val47_even.tfrecord")
test_ds = tf.data.TFRecordDataset(INFILE_prefix + "test47_even.tfrecord")

feat_spec = {
    'idx' : tf.io.FixedLenFeature([], tf.int64),
    'sentence' : tf.io.FixedLenFeature([], tf.string),
    'label' : tf.io.FixedLenFeature([], tf.int64)
}

# In[ ]:
def parse_ex(ex_proto):
    return tf.io.parse_single_example(ex_proto, feat_spec)

train_parsed = train_ds.map(parse_ex)
val_parsed = val_ds.map(parse_ex)
test_parsed = test_ds.map(parse_ex)

# In[ ]:
BATCH_SIZE = 22
EVAL_BATCH_SIZE = BATCH_SIZE * 2

# In[ ]:
with open('../data/lenghts/tweet_info.json') as json_file:
    data_info = json.load(json_file)

train_exs = data_info['train_length']
val_exs = data_info['val_length']
test_exs = data_info['test_length']

print((train_exs, val_exs, test_exs))

# In[ ]:
train_dataset = glue_convert_examples_to_features(examples = train_parsed,
                                                  tokenizer = tokenizer,
                                                  max_length = 128,
                                                  task = 'sst-2',
                                                  label_list=['0','-1','1']
                                                 )

val_dataset = glue_convert_examples_to_features(examples = val_parsed,
                                                  tokenizer = tokenizer,
                                                  max_length = 128,
                                                  task = 'sst-2',
                                                  label_list=['0','-1','1']
                                                 )

test_dataset = glue_convert_examples_to_features(examples = test_parsed,
                                                  tokenizer = tokenizer,
                                                  max_length = 128,
                                                  task = 'sst-2',
                                                  label_list=['0','-1','1']
                                                 )

# In[ ]:
train_dataset = train_dataset.shuffle(train_exs).batch(BATCH_SIZE).repeat(-1)
val_dataset = val_dataset.shuffle(val_exs).batch(BATCH_SIZE).repeat(-1)
test_dataset = test_dataset.batch(EVAL_BATCH_SIZE)

# In[ ]:
opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=opt, loss=loss, metrics=[metric])

# In[ ]:
train_steps = train_exs//BATCH_SIZE
val_steps = val_exs//EVAL_BATCH_SIZE
test_steps = test_exs//EVAL_BATCH_SIZE

# In[ ]:
history = model.fit(train_dataset,
                    steps_per_epoch=train_steps, 
                    epochs = 3, 
                    validation_data=val_dataset, 
                    validation_steps= val_steps, 
                    verbose = True, 
                    use_multiprocessing=True)

# In[ ]:
model.summary()

# In[ ]:
# This cell saves a model with a different name than what is included to avoid over writing.

# Change the path in the following cell to test out a new model
model.save_pretrained('../models/BERT-transfer1/')
tokenizer.save_vocabulary('../models/BERT-vocab1.dms')


# In[ ]:
new_tokenizer = BertTokenizer('../models/BERT-vocab1.dms') ### change here to try a new model
new_config = BertConfig.from_json_file('../models/BERT-config0.json')
new_model = TFBertForSequenceClassification.from_pretrained('../models/BERT-transfer1', config=new_config)

# In[ ]:
new_model.summary()

# In[ ]:
# model does not need to be compiled for prediction and confusion matrix production
# new_model.compile(optimizer=opt, loss=loss, metrics=[metric])
y_preds = model.predict(test_dataset, verbose = True, use_multiprocessing=True)

# In[ ]:
y_preds_sm = tf.nn.softmax(y_preds)
y_preds_argmax = tf.math.argmax(y_preds_sm, axis = 1)

# In[ ]:
y_true = tf.Variable([], dtype = tf.int64)

for feat, lab in test_dataset.take(-1):
    y_true = tf.concat([y_true, lab], 0)

# In[ ]:
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pandas
# get_ipython().magic(u'matplotlib inline')
# confusion = tf.math.confusion_matrix(y_true, y_preds_argmax).numpy()
# sns.heatmap(confusion, annot = True, fmt='g', cmap = plt.cm.Blues)
#plt.xlabel('Predicted label')
#plt.ylabel('True label')

# plt.show()

# classes 0, 1, 2 refer to labels 0, -1, 1 in this model. 
# this will be changed to make more sense i nthe future

# In[ ]:
# Exact Training Accuracy
m = tf.keras.metrics.Accuracy()
m.update_state(y_true, y_preds_argmax)

# In[ ]:
print(m.result().numpy())

# In[ ]:
