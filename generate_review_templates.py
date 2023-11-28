import json
from collections import OrderedDict


set_1 = []
set_2 = []
set_3 = []
set_4 = []
set_5 = []

def fill_set(file_path, set_):
    with open(file_path, "r", encoding='utf-8') as f:
        all_answers = json.load(f)
        for answer in all_answers[0:92]:
            ordered_answer = OrderedDict([
                ("question", answer["question"]),
                ("answer", answer["answer"]),
                ("evaluation", OrderedDict([
                    ("fluency", ""),
                    ("coherence", ""),
                    ("relevance", ""),
                    ("context_understanding", ""),
                    ("overall_quality", "")
                ]))
            ])
            set_.append(ordered_answer)


fill_set("test_results_new/answers/llama-2-7b-chat.Q5_K_M.gguf.json", set_1)
fill_set("test_results_new/answers/mistral-7b-instruct-v0.1.Q5_K_M.gguf.json", set_2)
fill_set("test_results_new/answers/mistral-7b-openorca.Q4_0.gguf.json", set_3)
fill_set("test_results_new/answers/zephyr-7b-alpha.Q5_K_M.gguf.json", set_4)
fill_set("test_results_new/answers/zephyr-7b-beta.Q5_K_M.gguf.json", set_5)

# reviewer_1-3 = llama-2-7b-chat.Q5_K_M.gguf.json
# reviewer_4-6 = mistral-7b-instruct-v0.1.Q5_K_M.gguf.json
# reviewer_7-9 = mistral-7b-openorca.Q4_0.gguf.json
# reviewer_10-12 = zephyr-7b-alpha.Q5_K_M.gguf.json
# reviewer_13-15 = zephyr-7b-beta.Q5_K_M.gguf.json

for i in range(1, 16):
    set_ = globals()[f'set_{(i-1)//3 + 1}']

    with open(f"review_templates/reviewer_{i}.txt", "w+", encoding='utf-8') as f:
        f.write(json.dumps(set_, indent=4))
