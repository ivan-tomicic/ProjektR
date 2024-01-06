import os
import json

dir = 'answers'
for filename in os.listdir(dir):
    file_path = os.path.join(dir, filename)
        
    with open(file_path, 'r') as file:
        #print(filename)
        llmAnswers = json.load(file)