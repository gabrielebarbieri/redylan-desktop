import spacy
from markovconstraints.markov_chain import MarkovProcess, parse_sequences
from markovconstraints.suffix_tree import get_suffix_tree
from datetime import datetime
from random import shuffle
import pickle


nlp = spacy.load("en_core_web_md")


def get_similarity(w1, w2):
    return nlp.vocab[w1.lower()].similarity(nlp.vocab[w2.lower()])


def tokenize(line):
    return nlp(line.strip().lower())


class Sentence(object):

    def __init__(self, words, orders):
        self.words = words
        self.orders = orders

    def __repr__(self):
        return ' '.join([w for w in self.words if w not in {'<s>', '</s>'}])


class Corpus:

    def __init__(self, source, order=3, language='en'):
        self.order = order
        self.language = language

        t = datetime.now()
        with open(source) as f:
            self.sentences = [tokenize(line) for line in f if line.strip()]
        self._words = set(word.text.lower() for sentence in self.sentences for word in sentence if not word.is_punct)

        print('time to tokenize {} sentences {}'.format(len(self.sentences), datetime.now() - t))

        t = datetime.now()
        to_parse = [[w.text.lower() for w in sentence if not w.is_punct] for sentence in self.sentences]
        self.matrices = parse_sequences(to_parse, order)
        print('time to estimate the matrices', datetime.now() - t)

        t = datetime.now()
        self.suffix_tree = get_suffix_tree(self.sentences)
        print('time to compute the suffix tree', datetime.now() - t)

    @property
    def words(self):
        return list(self._words)

    def generate_sentences(self, constraints, n=10):
        mp = MarkovProcess(self.matrices, constraints)
        sentences = []
        for _ in range(n):
            sequence = mp.generate()
            orders = self.suffix_tree.get_all_orders(sequence)
            sentences.append(Sentence(sequence, orders))
        return sentences

    def generate_semantic_sentence(self, sense, length, n, number__of_words=10):
        words = self.get_similar_words(sense, number__of_words)
        print(words)
        indices = [i for i in range(length)]
        shuffle(indices)
        for i in indices:
            try:
                cts = [None] * i + [words] + [None] * (length - i - 1)
                return self.generate_sentences(cts, n)
            except RuntimeError:
                pass
        return []

    def get_similar_words(self, sense, n=10):
        similarities = []
        for w in self.words:
            try:
                similarities.append((get_similarity(sense, w), w))
            except KeyError:
                pass
        return [k for _, k in sorted(similarities, reverse=True)[:n]]


dylan = Corpus('/Users/gabriele/Workspace/misc/redylan-desktop/redylan/data/dylan')
# with open('dylan.pkl', 'wb') as f:
#     pickle.dump(dylan, f)
for s in dylan.generate_semantic_sentence('love', 10, 10):
    print(s)
