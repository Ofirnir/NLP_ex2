import numpy as np
import simple_tagger as st
import  hmm_tagger as hmm
import nltk
# nltk.download('treebank')
from nltk.corpus import treebank

print(len(treebank.tagged_sents()))
#output: 3914
train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]
# print (train_data[0])
simple_tagger  = st.simple_tagger()
simple_tagger.train(train_data)
result_st = simple_tagger.evaluate(test_data)
print("Simple tagger : ")
print("Word level accuracy :" , result_st [0])
print("Sentence level accuracy :" , result_st [1])

hmm_tagger = hmm.hmm_tagger()
hmm_tagger.train(train_data)
hmm_tagger.evaluate(test_data)