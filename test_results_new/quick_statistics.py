import json

model_files = [
    "answers/llama-2-7b-chat.Q5_K_M.gguf.json",
    "answers/mistral-7b-instruct-v0.1.Q5_K_M.gguf.json",
    "answers/mistral-7b-openorca.Q4_0.gguf.json",
    "answers/zephyr-7b-alpha.Q5_K_M.gguf.json",
    "answers/zephyr-7b-beta.Q5_K_M.gguf.json"
]

metrics = ["fluency", "relevance", "coherence", "context_understanding", "overall_quality"]

stats = {
    model: {
        metric: {"num_of_grades": 0, "sum_of_grades": 0} for metric in metrics
    } for model in model_files
}

for model in model_files:
    with open(model, "r") as f:
        model_data = json.load(f)
        for obj_ in model_data:
            human_eval = obj_["evaluation"]["human_eval"]
            for metric, reviews in human_eval.items():
                for review in reviews.values():
                    try:
                        grade = float(review)
                        stats[model][metric]["num_of_grades"] += 1
                        stats[model][metric]["sum_of_grades"] += grade
                    except ValueError:
                        print(f"String {review} is not a valid grade for metric {metric}")
                        pass

for model, model_stats in stats.items():
    for metric, metric_stats in model_stats.items():
        print(f"Model: {model}, Metric: {metric}, Average grade: {metric_stats['sum_of_grades'] / metric_stats['num_of_grades']}")

    print(f"Average total: {sum([metric_stats['sum_of_grades'] / metric_stats['num_of_grades'] for metric_stats in model_stats.values()]) / len(model_stats)}")

    print("-" * 50)
    print("\n"*2)