import numpy as np
import simple_tagger as st
import  hmm_tagger as hmm
import pandas as pd
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
# print("Simple tagger : ")
# print("Word level accuracy :" , result_st [0])
# print("Sentence level accuracy :" , result_st [1])

hmm_tagger = hmm.hmm_tagger()
hmm_tagger.train(train_data)
result_hmm = hmm_tagger.evaluate(test_data)
#
# print("hmm tagger : ")
# print("Word level accuracy :" , result_hmm [0])
# print("Sentence level accuracy :" , result_hmm [1])

from nltk.tag import tnt
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)
# for sent in test_data:
data = []
# word_level_acc = 0.0
sentence_level_acc = 0.0
### Calc accuracy
for sentence in test_data:
    # current = []
    # correct_sentence = True
    words = [i[0] for i in sentence]
    labels = np.array( [j[1] for j in sentence])
    tagged = tnt_pos_tagger.tag(words)
    tagged = np.array([j[1] for j in tagged])
    predict = np.sum(tagged[tagged != 'Unk'] == labels[tagged != 'Unk'])
    if predict == len(tagged[tagged != 'Unk']):
        sentence_level_acc += 1
sentence_level_acc /= len(test_data)

memm = tnt_pos_tagger.evaluate(test_data), sentence_level_acc

final_table = np.empty([3, 2])
final_table = [[result_st[0], result_st[1]], [result_hmm[0], result_hmm[1]], [memm[0], memm[1]]]
rows = ["Simple tagger", "HMM tagger", "MEMM"]
columns = ["Word-level accuracy", "Sentence-level accuracy"]
df = pd.DataFrame(data=final_table, index=rows, columns=columns)
df.to_csv('final_table.csv')

