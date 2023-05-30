import numpy as np
from datasets import Dataset

data_pol = {'text': [], 'label': []}
data_subj = {'text': [], 'label': []}
with open('customds.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    l = line.strip()
    res = l.split('\t')
    data_pol['text'].append(res[0])
    data_subj['text'].append(res[0])
    aux = int(res[1])
    aux_s = int(res[2])
    data_pol['label'].append(aux)
    data_subj['label'].append(aux_s)

ds_pol = Dataset.from_dict(data_pol)
ds_pol = ds_pol.train_test_split(test_size=0.2, shuffle=True)

train_pol_labels = np.asarray(ds_pol['train']['label'])
test_pol_labels = np.asarray(ds_pol['test']['label'])
np.save('train_pol_labels', train_pol_labels)
np.save('test_pol_labels', test_pol_labels)

ds_pol.save_to_disk("sads_pol")

ds_subj = Dataset.from_dict(data_subj)
ds_subj = ds_subj.train_test_split(test_size=0.2, shuffle=True)

train_sub_labels = np.asarray(ds_subj['train']['label'])
test_sub_labels = np.asarray(ds_subj['test']['label'])
np.save('train_sub_labels', train_sub_labels)
np.save('test_sub_labels', test_sub_labels)

ds_subj.save_to_disk("sads_subj")

with open('xxxpoltrain.txt', 'w', encoding='utf-8') as file:
    for text in ds_pol['train']:
        prnt = text['text'].strip('.') + ' .'
        print(prnt, file=file)

with open('xxxpoltest.txt', 'w', encoding='utf-8') as file:
    for text in ds_pol['test']:
        prnt = text['text'].strip('.') + ' .'
        print(prnt, file=file)

with open('xxxsubtrain.txt', 'w', encoding='utf-8') as file:
    for text in ds_subj['train']:
        prnt = text['text'].strip('.') + ' .'
        print(prnt, file=file)

with open('xxxsubtest.txt', 'w', encoding='utf-8') as file:
    for text in ds_subj['test']:
        prnt = text['text'].strip('.') + ' .'
        print(prnt, file=file)
