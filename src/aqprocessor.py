from analyzer import Analyzer
from preprocessor import Preprocessor

class AQprocessor(object):

    def __init__(self, answers):
        self.answers = answers
        self.analyzer = Analyzer()
        self.preprocessor = Preprocessor()

    def process_answer(self, steps):
        texts = []
        texts_lemmatized = []

        for stepID, step in steps.items():
            text_lowercase = self.preprocessor.text_lowercase(step)
            text_segmented = self.preprocessor.text_segmentation(text_lowercase)
            text_stopword_removal = self.preprocessor.stopword_removal(text_segmented)
            text_lemmatized = self.preprocessor.lemmatization(text_stopword_removal)
            texts_lemmatized.append(text_lemmatized)

        texts_tfidf, dictionary = self.analyzer.calculate_tfidf(texts_lemmatized)
        texts_tfidf = [dict(text) for text in texts_tfidf]
        for text in texts_tfidf:
            temp = {}
            
            for id, val in text.items():
                temp[dictionary[id]] = val
                
            texts.append(temp)
        
        return texts_lemmatized, texts, dictionary

    def process_question(self, question):
        text_lowercase = self.preprocessor.text_lowercase(question)
        text_segmented = self.preprocessor.text_segmentation(text_lowercase)
        text_stopword_removal = self.preprocessor.stopword_removal(text_segmented)
        text_lemmatized = self.preprocessor.lemmatization(text_stopword_removal)

        return text_lemmatized
