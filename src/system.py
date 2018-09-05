from aqprocessor import AQprocessor
from analyzer import Analyzer
from collections import Counter
from data import answers

aqprocessor = AQprocessor(answers)
analyzer = Analyzer()
question = aqprocessor.process_question("When do I crack the egg into the pan?")

for methodID, steps in answers.items():
    texts_lemmatized = aqprocessor.process_answer(steps)

    vec1 = Counter(question)
    for text in texts_lemmatized:
        vec2 = Counter(text)
        cosine = analyzer.get_cosine(vec1, vec2)
        print(cosine)

    print('==================================================================================')
