import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = [ ("把 东 西 拿 给 我".split(), "Chinese"),
		("Give it to me".split(), "English"),
		("这 么 做 是 不 明 智 的".split(), "Chinese"),
		("No it is not a good idea to do it".split(), "English") ]

test_data = [ ("我 不 明 白 这 些 东 西".split(), "Chinese"),("it is lost on me".split(), "English")]

word_to_ix = {}
for sent, _ in data + test_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2