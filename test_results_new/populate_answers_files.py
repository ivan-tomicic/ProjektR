import json
import os

dict_ = {
    "reviewer_1": "llama-2-7b-chat.Q5_K_M.gguf.json",
    "reviewer_2": "llama-2-7b-chat.Q5_K_M.gguf.json",
    "reviewer_3": "llama-2-7b-chat.Q5_K_M.gguf.json",
    "reviewer_4": "mistral-7b-instruct-v0.1.Q5_K_M.gguf.json",
    "reviewer_5": "mistral-7b-instruct-v0.1.Q5_K_M.gguf.json",
    "reviewer_6": "mistral-7b-instruct-v0.1.Q5_K_M.gguf.json",
    "reviewer_7": "mistral-7b-openorca.Q4_0.gguf.json",
    "reviewer_8": "mistral-7b-openorca.Q4_0.gguf.json",
    "reviewer_9": "mistral-7b-openorca.Q4_0.gguf.json",
    "reviewer_10": "zephyr-7b-alpha.Q5_K_M.gguf.json",
    "reviewer_11": "zephyr-7b-alpha.Q5_K_M.gguf.json",
    "reviewer_12": "zephyr-7b-alpha.Q5_K_M.gguf.json",
    "reviewer_13": "zephyr-7b-beta.Q5_K_M.gguf.json",
    "reviewer_14": "zephyr-7b-beta.Q5_K_M.gguf.json",
    "reviewer_15": "zephyr-7b-beta.Q5_K_M.gguf.json",
}

def update_evaluations(json1, json2, reviewer):
    for item1 in json1:
        for item2 in json2:
            if item1["question"] == item2["question"]:
                item1["evaluation"]["human_eval"]["fluency"][reviewer] = item2["evaluation"]["fluency"]
                item1["evaluation"]["human_eval"]["coherence"][reviewer] = item2["evaluation"]["coherence"]
                item1["evaluation"]["human_eval"]["relevance"][reviewer] = item2["evaluation"]["relevance"]
                item1["evaluation"]["human_eval"]["context_understanding"][reviewer] = item2["evaluation"]["context_understanding"]
                item1["evaluation"]["human_eval"]["overall_quality"][reviewer] = item2["evaluation"]["overall_quality"]
                break

# Read JSON data from files
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


for reviewer, model_answers in dict_.items():
    if os.path.isfile(f"review_files/{reviewer}.json"):
        print(f"review_files/{reviewer}.json")
        review_file_data = read_json_file(f"review_files/{reviewer}.json")
        answers_file_data = read_json_file(f"answers/{model_answers}")
        update_evaluations(answers_file_data, review_file_data, reviewer)
        with open(f"answers/{model_answers}", 'w') as file:
            json.dump(answers_file_data, file, indent=4)
