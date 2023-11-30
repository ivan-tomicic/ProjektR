import json
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import LlamaCppEmbeddings


list_of_models = [
    {
        "model": "TheBloke/Llama-2-7b-Chat-GGUF",
        "model_file": "llama-2-7b-chat.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        "model_file": "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/Mistral-7B-OpenOrca-GGUF",
        "model_file": "mistral-7b-openorca.Q4_0.gguf"
    },
    {
        "model": "TheBloke/zephyr-7B-alpha-GGUF",
        "model_file": "zephyr-7b-alpha.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/zephyr-7B-beta-GGUF",
        "model_file": "zephyr-7b-beta.Q5_K_M.gguf"
    }

]




template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


prompt = PromptTemplate(template=template, input_variables=["context", "question"])


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'})

db = FAISS.load_local("faiss", embeddings)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})

def get_human_eval_metric(reviewer_from, reviewer_to):
    human_eval_metrics = {
        "fluency": {
            f"reviewer_{i}": "_" for i in range(reviewer_from, reviewer_to + 1)
        },
        "coherence": {
            f"reviewer_{i}": "_" for i in range(reviewer_from, reviewer_to + 1)
        },
        "relevance": {
            f"reviewer_{i}": "_" for i in range(reviewer_from, reviewer_to + 1)
        },
        "context_understanding": {
            f"reviewer_{i}": "_" for i in range(reviewer_from, reviewer_to + 1)
        },
        "overall_quality": {
            f"reviewer_{i}": "_" for i in range(reviewer_from, reviewer_to + 1)
        },
    }
    return human_eval_metrics


eval_dict = {
    "bleu_score": -100_000,
    "rouge_score": {
        "precision": -100_000,
        "recall": -100_000,
        "f1": -100_000
    },
    "diversity": -100_000,
}

with open("test_results_new/questions.json", "r", encoding='utf-8') as f:
    questions = json.load(f)

for i, model in enumerate(list_of_models):
    print("Starting model: ", model["model_file"])
    # create a file where we will store models answers
    answer_file = open(f"test_results_new/answers/{model['model_file']}.json", "w+", encoding='utf-8')
    llm = CTransformers(
        model=model['model'],
        model_file=model['model_file'],
        model_type="llama",
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt}
    )
    answers = []
    cnt_ = 1
    for question_dict in questions:
        print(f"Question {cnt_} out of {len(questions)}")
        cnt_ += 1
        question_number = list(question_dict.keys())[0]
        question = question_dict[question_number]
        time_start = time.time()
        output = qa.run(question)
        time_end = time.time()
        eval_dict["human_eval"] = get_human_eval_metric((i*3) + 1, (i+1)*3)
        answers.append({
            "question_number": question_number,
            "question": question,
            "answer": output,
            "time": round(time_end - time_start, 2),
            "evaluation": eval_dict,
        })
    answer_file.write(json.dumps(answers, indent=4))
