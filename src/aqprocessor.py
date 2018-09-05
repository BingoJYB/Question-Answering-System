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
            texts.append(text_stopword_removal)
            texts_lemmatized.append(text_lemmatized)

        # texts_tag = self.analyzer.calculate_tag(texts)
        # texts_tfidf = self.analyzer.calculate_tfidf(texts_lemmatized)
        return texts_lemmatized

    def process_question(self, question):
        text_lowercase = self.preprocessor.text_lowercase(question)
        text_segmented = self.preprocessor.text_segmentation(text_lowercase)
        text_stopword_removal = self.preprocessor.stopword_removal(text_segmented)
        text_lemmatized = self.preprocessor.lemmatization(text_stopword_removal)

        return text_lemmatized
