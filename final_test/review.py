import os
import json

dir = 'answers'
reviewer = {}

with open("questions.json", "r", encoding='utf-8') as f:
    questions = json.load(f)

r = 1
for i in range (1,31):
    q = 'q_'+str(i)
    reviewer[q] = {}
    reviewer[q]['question'] = questions[i-1]

    answers = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        
        with open(file_path, 'r') as file:
            #print(filename)
            llmAnswers = json.load(file)
        
        a = {}
        a['config'] = filename[:-5]
        a['answer'] = llmAnswers['answers'][i-1]['answer']
        a['time'] = llmAnswers['answers'][i-1]['time']
        a['human_eval'] = llmAnswers['answers'][i-1]['evaluation']['human_eval']
        answers.append(a)

    reviewer[q]['answers'] = answers

    if i%2 == 0:
        file = open(f"reviews/reviewer_{r}.json", "w+", encoding='utf-8')
        file.write(json.dumps(reviewer, indent=4))
        r += 1

        reviewer = {}