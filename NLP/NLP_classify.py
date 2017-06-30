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

class Classifier(nn.Module):
	def __init__(self, num_labels, vocab_size):
		super(Classifier, self).__init__()
		self.linear = nn.Linear(vocab_size, num_labels)
	def forward(self, bow_vec):
		return F.log_softmax(self.linear(bow_vec))

def make_bow_vector(sentence, word_to_ix):
	vec = torch.zeros(len(word_to_ix))
	for word in sentence:
		vec[word_to_ix[word]] += 1
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])


model = Classifier(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
	print(param)

sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)

label_to_ix = { "Chinese": 0, "English": 1 }

for instance, label in test_data:
	bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
	log_probs = model(bow_vec)
	print(log_probs)
print( next(model.parameters())[:,word_to_ix["西"]] )