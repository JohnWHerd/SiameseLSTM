from torch.utils.data import Dataset
from collections import Counter

class vocDS(Dataset):
    def __init__(self, df, sentences):
        self.q1 = df['question1'].tolist()
        self.q2 = df['question2'].tolist()
        self.labels = df['is_duplicate'].tolist()
        self.vocab = VocabMaker(sentences)

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        q1 = [self.vocab.start] + self.q1[index].split() + [self.vocab.end]
        q2 = [self.vocab.start] + self.q2[index].split() + [self.vocab.end]

        id1 = [self.vocab.word_id(word) for word in q1]
        id2 = [self.vocab.word_id(word) for word in q2]
        label = self.labels[index]

        return id1, id2, label

class VocabMaker(object):
    def __init__(self, sentences):

        self.pad = '<pad>'
        self.start= '<sos>'
        self.end= '<eos>' 
        self.unknown= '<unk>'

        id_words = [self.pad, self.start, self.end, self.unknown]

        counter = Counter()
        for sent in sentences:
            #eliminates any null or non-string values
            if type(sent) == str:
                counter.update(sent.split())
            
        counts = counter.items()
        counts = sorted(counts, key=lambda x: x[1], reverse=True)

        words = [c[0] for c in counts]

        id_words.extend(words)
        word_id = {w: i for i, w in enumerate(id_words)}

        self.id2word = id_words
        self.word2id = word_id

    def __len__(self):
        return len(self.id2word)
        
    def word_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.unknown
