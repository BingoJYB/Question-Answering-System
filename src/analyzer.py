import math

from gensim import corpora, models
from nltk.tag import pos_tag
from preprocessor import Preprocessor


class Analyzer(object):
    
    def __init__(self):
        self.preprocessor = Preprocessor()

    def calculate_tfidf(self, texts):

        texts_tfidf = []
        texts_word_tfidf = []

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        for doc in corpus_tfidf:
            texts_tfidf.append(doc)

        texts_tfidf = map(dict, texts_tfidf)
        dictionary = {id: word for word, id in dictionary.token2id.items()}

        for text in texts_tfidf:
            temp = {}

            for id, tfidf in text.items():
                temp[dictionary[id]] = tfidf

            texts_word_tfidf.append(temp)
        
        return texts_word_tfidf

    def get_tag(self, texts):

        texts_tag = []

        for text in texts:
            words, tags = zip(*pos_tag(text))
            words_lemmatized = self.preprocessor.lemmatization(words)
            text_tag = zip(words_lemmatized, tags)
            texts_tag.append(list(text_tag))

        return texts_tag

    def get_cosine(self, vec1, vec2, vec_tfidf):

        intersection = set(vec1.keys()) & set(vec2.keys())
        try:
            numerator = sum([vec1[x] * vec2[x] * vec_tfidf[x] for x in intersection])
        except:
            numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator
