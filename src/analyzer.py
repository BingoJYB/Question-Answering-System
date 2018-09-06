import math

from gensim import corpora, models
from operator import itemgetter

class Analyzer(object):

    def calculate_tfidf(self, texts):

        texts_with_tfidf = []

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        for doc in corpus_tfidf:
            texts_with_tfidf.append(doc)
            
        dictionary = {y : x for x, y in dictionary.token2id.items()}
        return texts_with_tfidf, dictionary

    def get_cosine(self, vec1, vec2, vec_tfidf):

        intersection = set(vec1.keys()) & set(vec2.keys())
        try:
            numerator = sum([vec1[x] * vec2[x] * vec_tfidf[x] for x in intersection])
        except:
            numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
