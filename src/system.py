from aqprocessor import AQprocessor
from analyzer import Analyzer
from collections import Counter
from data import answers

aqprocessor = AQprocessor(answers)
analyzer = Analyzer()
answer_tuples = {}
candidates = {}

for methodID, steps in answers.items():
    texts_lemmatized, texts_tag, vec_tfidf = aqprocessor.process_answer(steps)
    answer_tuples[methodID] = (texts_lemmatized, texts_tag, vec_tfidf)

while True:
    question = input('I: ')
    question_lemmatized, question_segmented = aqprocessor.process_question(question)
    question_tag = analyzer.get_tag([question_segmented])

    for methodID, tuples in answer_tuples.items():
        vec1 = Counter(question_lemmatized)
        candidates[methodID] = []

        for id, text in enumerate(tuples[0]):
            vec2 = Counter(text)
            cosine = analyzer.get_cosine(vec1, vec2, tuples[2][id])
            candidates[methodID].append(cosine)

        candidates[methodID] = sorted([(val, id) for id, val in enumerate(candidates[methodID])], reverse=True)[:3]
        best_candidates = aqprocessor.select_best_candidates(candidates[methodID], question, question_tag[0], tuples[1])
        print(best_candidates)
