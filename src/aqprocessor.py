from analyzer import Analyzer
from collections import Counter
from data import noun
from data import templates
from preprocessor import Preprocessor


class AQprocessor(object):

    def __init__(self, answers):
        self.answers = answers
        self.analyzer = Analyzer()
        self.preprocessor = Preprocessor()

    def process_answer(self, steps):
        texts_word_tfidf = []
        texts_tag = []
        texts_lemmatized = []

        for stepID, step in steps.items():
            text_lowercase = self.preprocessor.text_lowercase(step)
            text_segmented = self.preprocessor.text_segmentation(text_lowercase)
            text_stopword_removed = self.preprocessor.stopword_removal(text_segmented)
            text_lemmatized = self.preprocessor.lemmatization(text_stopword_removed)
            texts_tag.append(text_segmented)
            texts_lemmatized.append(text_lemmatized)

        texts_tag = self.analyzer.get_tag(texts_tag)
        texts_tfidf, dictionary = self.analyzer.calculate_tfidf(texts_lemmatized)
        texts_tfidf = [dict(text) for text in texts_tfidf]

        for text in texts_tfidf:
            temp = {}

            for id, tfidf in text.items():
                temp[dictionary[id]] = tfidf

            texts_word_tfidf.append(temp)

        return texts_lemmatized, texts_tag, texts_word_tfidf

    def process_question(self, question):
        question_lowercase = self.preprocessor.text_lowercase(question)
        question_segmented = self.preprocessor.text_segmentation(question_lowercase)
        question_stopword_removed = self.preprocessor.stopword_removal(question_segmented)
        question_lemmatized = self.preprocessor.lemmatization(question_stopword_removed)
        question_tag = self.analyzer.get_tag([question_segmented])

        return question_lemmatized, question_tag

    def select_best_candidates(self, answer, question):

        original_question = self.preprocessor.text_lowercase(question[0])
        question_vec = Counter(question[1])
        candidates = []
        best_candidates = set()

        for id, text in enumerate(answer[0]):
            candidate_vec = Counter(text)
            cosine = self.analyzer.get_cosine(question_vec, candidate_vec, answer[2][id])
            candidates.append(cosine)

        candidates = sorted([(val, id) for id, val in enumerate(candidates)],
                            reverse=True)[:3]

        for key, val in templates.items():
            if key in original_question:
                for candidate in candidates:
                    text_tag = answer[1][candidate[1]]

                    for qtag in question[2][0]:
                        if qtag[1] in noun:
                            for id, ttag in enumerate(text_tag):
                                if ttag[0] == qtag[0]:
                                    if id > 2:
                                        combined = text_tag[id - 3:id] + text_tag[id + 1:id + 3]
                                    else:
                                        combined = text_tag[:id] + text_tag[id + 1:id + 3]
                                    for tp in combined:
                                        if tp[1] in val:
                                            best_candidates.add(candidate)

                break

        return best_candidates
