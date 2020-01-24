#!/usr/bin/env python
# coding: utf-8

# The sections that are commented out were left in to allow for the creation of an uneven, but larger, dataset.

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from sklearn.model_selection import train_test_split

import json

# CHANGE THESE TO TRAINING DATA FILEPATH AND OUT-FILEPATH
prelabeled_tweets = '../data/prelabeled/tweets_47k.csv'

OUTFILE_prefix = '../data/prelabeled/'


# In[2]:


DF = pd.read_csv(prelabeled_tweets)
DF.shape


# In[3]:


DF.head()


# In[4]:


DF.Stance.unique()
DF.Stance = DF.Stance.astype('int32', copy = False)


# In[5]:


def clean_tweets():
    '''
    Takes the DF defined above and (in this order) applies the following preprocessing steps:
    1. Remove cases
    2. Replaces and URL's with "LINK"
    3. Replaces any twitter handels with "USERNAME"
    4. Removes any punctuation
    
    Note: Stop words will not be removed in this iteration because they may add some information.
    '''
    # Remove cases from the tweets
    DF.Tweet = DF.Tweet.str.lower()
    
    # Remove URL links
    DF.Tweet = DF.Tweet.str.replace('http\S+|www.\S+', 'LINK', case = False)
    
    # Remove usernames
    DF.Tweet = DF.Tweet.str.replace('@.*w', 'USERNAME ', case = False)
    
    # Remove #'s? - Uncomment next line if you aren't using the next filter
#     DF.Tweet = DF.Tweet.str.replace('#', '', case = False)
    
    # Remove remaining punctuation
    DF.Tweet = DF.Tweet.str.replace('[^\w\s]', '')
    
    # Convert Stance to a numerical val - Alread done for current DF
    # stances = {'NONE':0, 'AGAINST':-1, 'FAVOR':1}
    # DF.Stance =DF.Stance.map(stances)
    # DF.astype({'Stance': 'int32'}, copy = False)
    
clean_tweets()


# In[6]:


print(f"0's: {(DF.Stance == 0).sum()}")
print(f"1's: {(DF.Stance == 1).sum()}")
print(f"-1's: {(DF.Stance == -1).sum()}")


# In[7]:


print(DF.Stance.shape)
print(DF.dtypes)


# In[8]:


# Sampling 6247 from each label
df_pos = DF[DF.Stance == 1].sample(6247, replace = False)
df_neu = DF[DF.Stance == 0].sample(6247, replace = False)
df_neg = DF[DF.Stance == -1].sample(6247, replace = False)
print(df_pos.shape, df_neu.shape, df_neg.shape)


# In[9]:


df = pd.concat([df_pos, df_neu, df_neg])
print(df.shape)


# In[11]:


# Make All
# X_train, X_test, y_train, y_test = train_test_split(DF.Tweet, DF.Stance, test_size = .2, shuffle = True)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, shuffle = True)


# Make evenly classed subsample 
X_train, X_test, y_train, y_test = train_test_split(df.Tweet, df.Stance, test_size = .2, shuffle = True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, shuffle = True)


# In[12]:


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)


# In[13]:


train = pd.DataFrame(np.array([X_train, y_train]).T)
test = pd.DataFrame(np.array([X_test, y_test]).T)
val = pd.DataFrame(np.array([X_val, y_val]).T)


# In[14]:


train


# In[15]:


train.shape


# In[ ]:


train_csv = train.values
test_csv = test.values
val_csv = val.values


# In[ ]:


def make_tf_ex(feats, lab):
    tf_ex = tf.train.Example(features = tf.train.Features(feature= {
        'idx' : tf.train.Feature(int64_list = tf.train.Int64List(value = [feats[0]])),
        'sentence' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [feats[1].encode('utf-8')])),
        'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [lab]))
    }))
    
    return tf_ex


# In[ ]:


def convert_csv_to_tf_record(csv, file_name):
    writer = TFRecordWriter(file_name)
    for index,row in enumerate(csv):
        try:
            if row is None:
                print("row was None")
                raise Exception('Row Missing')
                
            if row[0] is None or row[1] is None:
                print("row[0] or row[1] was None")
                raise Exception('Value Missing')
                
            if row[0].strip() is '':
                print("row[0].strip() was ''")
                raise Exception('Utterance is empty')
                
            feats = (index, row[0])
            lab = row[1]
            example = make_tf_ex(feats, lab)
            writer.write(example.SerializeToString())

        except Exception as inst:
            print(type(inst))
            print(Exception.args)
            print(Exception.with_traceback)
            
    writer.close()

def generate_json_info(local_file_name):
    info = {"train_length": len(train),
            "val_length": len(val),
            "test_length": len(test)}

    with open(local_file_name, 'w') as outfile:
        json.dump(info, outfile)


# In[ ]:


# Make All

# convert_csv_to_tf_record(train_csv, "data/train_large.tfrecord")
# convert_csv_to_tf_record(test_csv, "data/test_large.tfrecord")
# convert_csv_to_tf_record(val_csv, "data/val_large.tfrecord")

# Make even subsample - ~18,000 in total
convert_csv_to_tf_record(train_csv, OUTFILE_prefix + "train47.tfrecord")
convert_csv_to_tf_record(test_csv, OUTFILE_prefix + "test47.tfrecord")
convert_csv_to_tf_record(val_csv, OUTFILE_prefix + "val47.tfrecord")


# In[ ]:


generate_json_info("../data/lengths/tweet47_info.json")


# In[ ]:




