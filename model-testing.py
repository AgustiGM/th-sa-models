import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import classification_report
from tensorflow import keras
from transformers import AutoTokenizer, BertTokenizer
import keras.losses


def dummy_loss(a, b):
    return


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1)


lss = keras.losses.binary_crossentropy

dataset = load_dataset("sst", "default")
dataset = dataset['test']
labels = np.array(dataset["label"][0:15])
model = tf.keras.models.load_model(os.getcwd() + os.sep + "test", custom_objects={"lss": lss})
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
tokenized_data = tokenizer(dataset["sentence"][0:15], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)
labels = np.array(dataset["label"][0:15])
# loss, accuracy, f1_score, precision, recall = model.evaluate(tokenized_data, labels, batch_size=128)
y_pred = model.predict(tokenized_data, batch_size=64, verbose=1)
y_pred = softmax(y_pred['logits'])
print(classification_report(labels[0:15], y_pred))
# print(f'loss: {loss}')
# print(f'accuracy: {accuracy}')
# print(f'f1_score: {f1_score}')
# print(f'precision: {precision}')
# print(f'recall: {recall}')
