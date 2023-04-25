import os
import shutil

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from datasets import load_dataset
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification

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

tokenized_data["train"].set_format(type='tf', columns=['attention_mask', 'input_ids', 'token_type_ids', 'label'])
# rename label as labels, as expected by tf models
tokenized_data["train"].rename_column('label', 'labels')
# set format
tokenized_data["validation"].set_format(type='tf', columns=['attention_mask', 'input_ids', 'token_type_ids', 'label'])
# make the renaming
tokenized_data["validation"].rename_column('label', 'labels')

# convert to TF dataset
# train set
train_data = tokenized_data["train"].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'token_type_ids'],
    label_cols=['labels'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8
)
# validation set
val_data = tokenized_data["validation"].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'token_type_ids'],
    label_cols=['labels'],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8
)

EPOCHS = 3
TRAINING_STEPS = len(train_data) * EPOCHS

# define a learning rate scheduler
lr_scheduler = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=5e-5,
    end_learning_rate=0.0,
    decay_steps=TRAINING_STEPS
)

# define an Adam optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_scheduler)

# define loss function
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# build model

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)

# compile model
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# What layers, parameters are there in the model
print(model.summary())

hist = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

history = hist.history
plt.plot(range(1, 4), history["accuracy"], label='train')
plt.plot(range(1, 4), history['val_accuracy'], label='val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.xticks([1, 2, 3])
plt.legend()
plt.show()

model.save("./saved_model")
