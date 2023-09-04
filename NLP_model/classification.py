import torchtext
import os
import collections
os.makedirs('NLP_model', exist_ok= True)
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='NLP_model')
classes = ['World', 'Sport', 'Business', 'Sci/Tech']

train_dataset = list(train_dataset)
test_dataset = list(test_dataset)

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# for i,x in zip(range(5), train_dataset):
#     print(f'**{classes[x[0]]}** -> {x[1]}\n')

first_sentence = train_dataset[0][1]
second_sentence = train_dataset[1][1]

f_tokens = tokenizer(first_sentence)
s_tokens = tokenizer(second_sentence)

counter = collections.Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))
vocab = torchtext.vocab.Vocab(counter)

word_lookup = [list((vocab[w], w)) for w in f_tokens]

word_lookup = [list((vocab[w], w)) for w in s_tokens]

vocab_size = len(vocab)
print(f"Vocab size if {vocab_size}")

def encode(x):
    return [vocab.get_stoi() for s in tokenizer(x)]

vec = encode(first_sentence)
print(vec)

def decode(x):
    return [vocab.itos[i] for i in x]

decode(vec)