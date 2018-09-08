from analyzer import Analyzer
from collections import Counter
from data import templates
from preprocessor import Preprocessor


class AQprocessor(object):

    def __init__(self, answers):
        
        self.answers = answers
        self.analyzer = Analyzer()
        self.preprocessor = Preprocessor()

    # process the answer to get lemmatized words, part of speech, term frequency-inverse document frequency
    def process_answer(self, steps):
        
        texts_tag = []              # list of (word, part of speech) tuples
        texts_lemmatized = []       # list of lemmatized words

        for stepID, step in steps.items():
            text_lowercase = self.preprocessor.text_lowercase(step)
            text_segmented = self.preprocessor.text_segmentation(text_lowercase)
            text_stopword_removed = self.preprocessor.stopword_removal(text_segmented)
            text_lemmatized = self.preprocessor.lemmatization(text_stopword_removed)
            texts_tag.append(text_segmented)
            texts_lemmatized.append(text_lemmatized)

        texts_tag = self.analyzer.get_tag(texts_tag)
        texts_word_tfidf = self.analyzer.calculate_tfidf(texts_lemmatized)

        return texts_lemmatized, texts_tag, texts_word_tfidf

    # process the question to get lemmatized question, part of speech
    def process_question(self, question):
        
        question_lowercase = self.preprocessor.text_lowercase(question)
        question_segmented = self.preprocessor.text_segmentation(question_lowercase)
        question_stopword_removed = self.preprocessor.stopword_removal(question_segmented)
        question_lemmatized = self.preprocessor.lemmatization(question_stopword_removed)
        question_tag = self.analyzer.get_tag([question_segmented])

        return question_lemmatized, question_tag

    # select best candidates from candidates
    def select_best_candidates(self, answer, question):

        candidates = []
        best_candidates = set()
        question_vec = Counter(question[1])

        for id, text in enumerate(answer[0]):
            candidate_vec = Counter(text)
            cosine = self.analyzer.get_cosine(question_vec, candidate_vec, answer[2][id])
            candidates.append(cosine)

        candidates = sorted([(val, id) for id, val in enumerate(candidates)],
                            reverse=True)[:3]

        for key, tuple in templates.items():
            if key in question[0].lower():
                for candidate in candidates:
                    text_tag = answer[1][candidate[1]]

                    for qtag in question[2][0]:
                        if qtag[1] in ['NN', 'NNS']:
                            for id, ttag in enumerate(text_tag):
                                if ttag[0] == qtag[0]:
                                    if id > 2:
                                        combined = text_tag[id-3:id] + text_tag[id+1:id+4]
                                    else:
                                        combined = text_tag[:id] + text_tag[id+1:id+4]
                                    for tp in combined:
                                        if tp[0] in tuple['keyword'] or tp[1] in tuple['tag']:
                                            best_candidates.add(candidate)

                break

        return best_candidates
