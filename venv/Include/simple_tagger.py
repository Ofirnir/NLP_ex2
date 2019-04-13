from collections import defaultdict,Counter

class simple_tagger:
    def __init__(self) :
        self.mapping = defaultdict()

    def train(self,data):
        counter = Counter()
        ## Count words
        for sentence in data:
            for word in sentence:
                counter[word] += 1
        ### iterating over ordered list and adding the most commons tags
        for count in counter.most_common():
            tag = count[0]
            if tag[0] not in  self.mapping:
                self.mapping[tag[0]] = tag[1]
        return self.mapping

    def evaluate(self,data):
        count = 0
        word_level_acc = 0.0
        sentence_level_acc =0.0
        for sentence in data:
            correct_sentence = True
            for word in sentence:
                count += 1
                if word[0]  in self.mapping:
                    if self.mapping[word [0]] == word[1] : word_level_acc += 1
                    else :
                        correct_sentence = False
                else:
                    self.mapping[word[0]] = sentence[ count % len(sentence) ][1] #assign random tag
            if correct_sentence:  sentence_level_acc += 1

        return word_level_acc / count , sentence_level_acc /len(data)
