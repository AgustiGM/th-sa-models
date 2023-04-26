import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from datasets import load_dataset
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification, AutoModel, \
    TFAutoModel, TFBertModel

tf.get_logger().setLevel('ERROR')

dataset = load_dataset("SetFit/sst5", "default")
train_dataset = dataset['train']
test_dataset = dataset['test']

checkpoint = 'bert-base-uncased'
# bert tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors='tf')
# data collator for dynamic padding as per batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')


def tokenize_function(example):
    return tokenizer(example['text'], truncation=True)


tokenized_data = dataset.map(tokenize_function, batched=True)

# set format
tokenized_data['test'].set_format(type='tf',
                                  columns=['input_ids', 'token_type_ids', 'attention_mask', 'label']
                                  )
# rename label as labels
tokenized_data['test'].rename_column('label', 'labels')
y_true = tokenized_data['test']['label'].numpy()
# convert to TF Dataset
test_data = tokenized_data["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask'],
    label_cols=['labels'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8
)
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = TFAutoModelForSequenceClassification.from_pretrained("./saved_model")

print(model.summary())
preds = model.predict(test_data, verbose=1)['logits']

label_preds = np.argmax(preds, axis=1)

l = len(y_true)
acc = sum([label_preds[i] == y_true[i] for i in range(l)]) / l
print(acc)
