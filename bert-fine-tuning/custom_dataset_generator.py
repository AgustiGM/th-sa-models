from datasets import Dataset

data = {'text': [], 'label': []}
with open('customds.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    l = line.strip()
    res = l.split('\t')
    data['text'].append(res[0])
    aux = int(res[1])
    # if aux < 2:
    #     aux = 0
    # elif aux == 2:
    #     aux = 1
    # else:
    #     aux = 2

    data['label'].append(aux)

ds = Dataset.from_dict(data)
ds = ds.train_test_split(test_size=0.2, shuffle=True)
ds.save_to_disk("sads")
print(ds)
