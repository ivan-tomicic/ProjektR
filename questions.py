import json
import random
from typing import Iterable


def get_question_tuples(file: str) -> Iterable[tuple[str, str]]:
    with open(file, "rb") as f:
        queries = json.load(f)
        return map(lambda q: (list(q.keys())[0], list(q.values())[0]), queries)


def get_random_question(file: str) -> tuple[str, str]:
    with open(file, "rb") as f:
        queries = json.load(f)
        query = random.choice(queries)
        return list(query.keys())[0], list(query.values())[0]
