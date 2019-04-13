from collections import defaultdict,Counter
import numpy as np
import viterbi as vt


class hmm_tagger:
    def __init__(self):
        self.A = None
        self.B = None
        self.Pi = None
        self.unique_words = defaultdict()
        self.unique_labels = defaultdict()

    def train(self, data):
        count = 0
        word_id = 0
        label_id = 0
        A_counter = Counter()
        B_counter = Counter()
        for sentence in data:
            prev_label= 0
            for w in range(len(sentence)):
                count += 1
                word = sentence[w][0]
                label = sentence[w][1]
                # Mapping unique words to a serial number
                if word not in self.unique_words:
                    self.unique_words[word] = word_id
                    word = word_id
                    word_id += 1
                else:
                    word = self.unique_words[word]
                # Mapping unique labels to a serial number
                if label not in self.unique_labels:
                    self.unique_labels[label] = label_id
                    label = label_id
                    label_id += 1
                else:
                    label = self.unique_labels[label]

                if w != 0:
                    A_counter[(prev_label, label)] += 1

                B_counter[(label,word)] += 1
                prev_label = label

        self.A = np.zeros(shape=(label_id,label_id))
        self.B = np.zeros(shape=(label_id , word_id))
        self.Pi = np.zeros(shape=label_id)

        for c in B_counter:
            label = c[0]
            word = c[1]
            self.B[label][word] = B_counter[c]

        for c in A_counter:
            i = c[0]
            j = c[1]
            self.A[i, j] = A_counter[i, j]

        for i in range(label_id):
            self.Pi[i] = sum(self.A[i])
            self.A[i] /= sum(self.A[i])
            self.B[i] /= sum(self.B[i])

        self.Pi /= sum(self.Pi)

    def comparisons(self, begin, end, sentence, seq):
        correct_sentence = True
        correct = 0.0
        index = 0
        for i in range(begin, end):
            label = self.unique_labels[sentence[i][1]]
            if (label != seq[index]):
                correct_sentence = False
            else:
                correct += 1
            index += 1
        return (correct, correct_sentence)

    def evaluate(self, data):
        word_level_acc = 0.0
        sentence_level_acc = 0.0
        count_words = 0
        for sentence in data:
            word_list = []
            begin_index = 0
            correct_sentence = True
            for i in range(len(sentence)):
                count_words += 1
                word = sentence[i][0]
                label = sentence[i][1]
                if word not in self.unique_words:
                    seq = vt.viterbi(word_list, self.A, self.B, self.Pi)
                    word_list.clear()
                    word_acc, sentence_acc = self.comparisons(begin_index, i, sentence, seq)
                    word_level_acc += word_acc
                word_list[i] = self.unique_words[word]
            last_index = len(sentence)
            seq = vt.viterbi(word_list, self.A, self.B, self.Pi)





