import math

from gensim import corpora, models, similarities
from operator import itemgetter
from nltk.tag import pos_tag

class Analyzer(object):

    def calculate_tfidf(self, texts):

        texts_with_tfidf = []

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        for doc in corpus_tfidf:
            texts_with_tfidf.append(sorted(doc, key=itemgetter(1), reverse=True)[:10])

        return texts_with_tfidf

    def calculate_tag(self, texts):

        texts_with_tag = []

        for text in texts:
            texts_with_tag.append(pos_tag(text))

        return texts_with_tag

    def get_cosine(self, vec1, vec2):

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
