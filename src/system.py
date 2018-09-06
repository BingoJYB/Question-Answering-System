from aqprocessor import AQprocessor
from analyzer import Analyzer
from collections import Counter
from data import answers

aqprocessor = AQprocessor(answers)
analyzer = Analyzer()
question = aqprocessor.process_question("How long do I cook the noodles?")

for methodID, steps in answers.items():
    texts_lemmatized, vec_tfidf, dictionary = aqprocessor.process_answer(steps)

    vec1 = Counter(question)
    for id, text in enumerate(texts_lemmatized):
        vec2 = Counter(text)
        cosine = analyzer.get_cosine(vec1, vec2, vec_tfidf[id])
        print(cosine)

    print('==================================================================================')
