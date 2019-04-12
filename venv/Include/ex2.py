import numpy as np
import simple_tagger as st
import nltk
nltk.download('treebank')
from nltk.corpus import treebank
print(len(treebank.tagged_sents()))
#output: 3914
train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]
# print (train_data[0])
x = st.simple_tagger()
x.train(train_data)
print(x.evaluate(test_data))
