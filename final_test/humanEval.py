import os
import json

dir1 = 'reviews'
dir2 = 'answers'

for filename in os.listdir(dir1):
    file_path = os.path.join(dir1, filename)
    print(filename) 
    with open(file_path, 'r') as file:
        review = json.load(file)

    for i, q in review.items():
        for ans in q['answers']:
            #print(ans['config'])
            filename = ans['config'] + '.json'
            file_path = os.path.join(dir2, filename)
            with open(file_path, 'r') as file:
                conf = json.load(file)
            #print(conf['answers'][int(i[2:])-1]['question_number'])
            #print(i)
            if (i != conf['answers'][int(i[2:])-1]['question_number']):
                print('Greska')
            conf['answers'][int(i[2:])-1]['evaluation']['human_eval'] = ans['human_eval']
            with open(file_path, 'w') as file:
                json.dump(conf, file, indent=4)
    


'''
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
        '''