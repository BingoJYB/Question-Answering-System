from analyzer import Analyzer
from data import noun
from data import templates
from preprocessor import Preprocessor


class AQprocessor(object):

    def __init__(self, answers):
        self.answers = answers
        self.analyzer = Analyzer()
        self.preprocessor = Preprocessor()

    def process_answer(self, steps):
        vec_tfidf = []
        texts_tag = []
        texts_lemmatized = []

        for stepID, step in steps.items():
            text_lowercase = self.preprocessor.text_lowercase(step)
            text_segmented = self.preprocessor.text_segmentation(text_lowercase)
            text_stopword_removal = self.preprocessor.stopword_removal(text_segmented)
            text_lemmatized = self.preprocessor.lemmatization(text_stopword_removal)
            texts_tag.append(text_segmented)
            texts_lemmatized.append(text_lemmatized)

        texts_tag = self.analyzer.get_tag(texts_tag)
        texts_tfidf, dictionary = self.analyzer.calculate_tfidf(texts_lemmatized)
        texts_tfidf = [dict(text) for text in texts_tfidf]
        for text in texts_tfidf:
            temp = {}

            for id, val in text.items():
                temp[dictionary[id]] = val

            vec_tfidf.append(temp)

        return texts_lemmatized, texts_tag, vec_tfidf

    def process_question(self, question):
        question_lowercase = self.preprocessor.text_lowercase(question)
        question_segmented = self.preprocessor.text_segmentation(question_lowercase)
        question_stopword_removal = self.preprocessor.stopword_removal(question_segmented)
        question_lemmatized = self.preprocessor.lemmatization(question_stopword_removal)

        return question_lemmatized, question_segmented

    def select_best_candidates(self, candidates, question, question_tag, texts_tag):

        question = self.preprocessor.text_lowercase(question)
        best_candidates = set()

        for key, val in templates.items():
            if key in question:
                for candidate in candidates:
                    text_tag = texts_tag[candidate[1]]

                    for qtag in question_tag:
                        if qtag[1] in noun:
                            for id, ttag in enumerate(text_tag):
                                if ttag[0] == qtag[0]:
                                    if id > 2:
                                        combined = text_tag[id-3:id] + text_tag[id+1:id+3]
                                    else:
                                        combined = text_tag[:id] + text_tag[id+1:id+3]
                                    for tp in combined:
                                        if tp[1] in val:
                                            best_candidates.add(candidate)

                break

        return best_candidates
