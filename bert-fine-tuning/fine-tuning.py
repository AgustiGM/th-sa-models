import sys

import keras.losses
import numpy as np
from keras.optimizers import Adam

from transformers import BertTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

dataset = load_dataset("sst", "default")
dataset = dataset['train']
# sep = os.sep
# gs_folder_bert = f'D:{sep}GEI{sep}Quart{sep}Primavera23{sep}TFG{sep}code{sep}bert-large-uncased'
#
#
# sys.exit(0)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_data = tokenizer(dataset["sentence"][0:150], return_tensors="np", padding=True)
tokenized_data = dict(tokenized_data)
labels = np.array(dataset["label"][0:150])

lss = keras.losses.binary_crossentropy

model = TFAutoModelForSequenceClassification.from_pretrained("bert-large-uncased")
model.compile(optimizer=Adam(3e-5), loss=lss)

model.fit(tokenized_data, labels)

model.save(os.getcwd()+os.sep+"test", )
# model.save("mymodel.h5")
