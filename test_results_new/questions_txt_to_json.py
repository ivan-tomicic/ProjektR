import json
import re


def convert_to_json(text):

    pattern = re.compile(r'(\d+)\.\s(.*?)\s*(\*+)?$')
    lines = text.split('\n')
    questions = []

    for line in lines:
        match = pattern.match(line)
        if match:
            number = match.group(1)
            question_text = match.group(2)
            difficulty = match.group(3) if match.group(3) else ""

            questions.append({"q_" + number: question_text, "difficulty": difficulty})

    json_output = json.dumps(questions, indent=3)
    return json_output

with open("pitanja.txt", "r", encoding='utf-8') as f:
    text = f.read()

# json_output = convert_to_json(text)

