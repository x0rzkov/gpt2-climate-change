#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import json
from sklearn.model_selection import train_test_split

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features

from transformers.configuration_bert import BertConfig


# In[ ]:


tokenizer = BertTokenizer('../models/BERT-vocab1.dms')

config = BertConfig.from_json_file('../models/BERT-config0.json')

model = TFBertForSequenceClassification.from_pretrained('../models/BERT-transfer1', config=config)


# In[ ]:


fname = '../data/prelabeled/test47_even.tfrecord'
# BATCH_SIZE = 64
feat_spec = {
    'idx' : tf.io.FixedLenFeature([], tf.int64),
    'sentence' : tf.io.FixedLenFeature([], tf.string),
    'label' : tf.io.FixedLenFeature([], tf.int64)
}

def parse_ex(ex_proto):
    return tf.io.parse_single_example(ex_proto, feat_spec)

tweets = tf.data.TFRecordDataset(fname)
tweets = tweets.map(parse_ex)

# with open('data/tweet_info.json')as j_file:
#     data_info = json.load(j_file)
#     num_samples = data_info['DF_length']

eval_df = glue_convert_examples_to_features(examples = tweets,
                                            tokenizer = tokenizer,
                                            max_length = 128,
                                            task = 'sst-2',
                                            label_list = ['0','-1', '1'])


# In[ ]:


eval_df = eval_df.batch(64)


# In[ ]:


y_preds = model.predict(eval_df, verbose = True, use_multiprocessing=True)


# In[ ]:


# y_preds_sm = tf.nn.softmax(y_preds)
# y_preds_argmax = tf.math.argmax(y_preds_sm, axis = 1)

y_true = tf.Variable([], dtype = tf.int64)

for feat, lab in eval_df.take(-1):
    y_true = tf.concat([y_true, lab], 0)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pandas
# get_ipython().magic(u'matplotlib inline')
# confusion = tf.math.confusion_matrix(y_true, y_preds_argmax).numpy()

# sns.heatmap(confusion, annot = True, fmt='g', cmap = plt.cm.Blues, )
# plt.xlabel('Predicted label')
# plt.ylabel('True label')

# plt.show()

# classes 0, 1, 2 refer to labels 0, -1, 1 in this model. 
# this will be changed to make more sense i nthe future


# In[ ]:


# from sklearn.metrics import classification_report
# print(classification_report(y_true, y_preds_argmax))


# In[ ]:


import matplotlib
matplotlib.__version__


# In[ ]:


np.savetxt('../data/prelabeled/test47_predicted_labels.csv', y_preds_argmax.numpy(), delimiter = ',')


# # From saved labels

# In[ ]:


Y_true_numpy = y_true.numpy()


# In[ ]:


y_pred_labels = np.loadtxt('../data/prelabeled/test47_predicted_labels.csv', delimiter=',')


# In[ ]:


y_pred_labels


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
get_ipython().magic(u'matplotlib inline')
confusion = tf.math.confusion_matrix(Y_true_numpy, y_pred_labels).numpy()
confusion = confusion/confusion.sum()
df_confusion = pd.DataFrame(confusion, index = ['Neutral', 'Positive', 'Deny'], 
                            columns = ['Neutral', 'Positive', 'Deny'])


sns.heatmap(df_confusion, annot = True, fmt='g', cmap = plt.cm.Blues, )
plt.xlabel('Predicted label', weight = 'bold')
plt.ylabel('True label', weight = 'bold')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(Y_true_numpy, y_pred_labels))


# In[ ]:




