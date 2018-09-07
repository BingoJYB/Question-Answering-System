from aqprocessor import AQprocessor
from data import answers

aqprocessor = AQprocessor(answers)
candidates = {}

for methodID, steps in answers.items():
    texts_lemmatized, texts_tag, texts_word_tag = aqprocessor.process_answer(steps)
    answers[methodID] = (texts_lemmatized, texts_tag, texts_word_tag)

while True:
    question = input('Q: ')

    if question[-1] != '?':
        print('A: not a valid question')

    else:
        question_lemmatized, question_tag = aqprocessor.process_question(question)
        question = (question, question_lemmatized, question_tag)

        for methodID, answer in answers.items():
            candidates[methodID] = aqprocessor.select_best_candidates(answer, question)

        output = ''
        for methodID, candidate in candidates.items():
            steps = ''

            for step in sorted(candidate, key=lambda x: x[1]):
                steps = steps + str(step[1] + 1) + '&'

            if len(candidate) > 0:
                output = output + 'step ' + steps[:-1] + ' in ' + methodID + ', '

        if output == '':
            print('A: not contained')

        else:
            print('A: ' + output[:-2])
