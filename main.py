import json
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from accelerate import Accelerator
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from  langchain.schema.language_model import BaseLanguageModel
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
torch.cuda.empty_cache()


list_of_models = [
    {
        "model": "TheBloke/Mistral-7B-OpenOrca-GGUF",
        "model_file": "mistral-7b-openorca.Q4_0.gguf"
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

accelerator = Accelerator()

with open("test_results_new/questions.json", "r", encoding='utf-8') as f:
    questions = json.load(f)

for i, model in enumerate(list_of_models):
    print("Starting model: ", model["model_file"])
    # create a file where we will store models answers
    #answer_file = open(f"test_results_new/answers/{model['model_file']}.json", "w+", encoding='utf-8')
    config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'context_length': 4000, 'temperature': 0.01,
              'gpu_layers': 25}
    llm = CTransformers(
        model=model['model'],
        model_file=model['model_file'],
        model_type="llama",
        gpu_layers=25,
        config=config,
    )
    llm, config = accelerator.prepare(llm, config)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt},
        return_source_documents=True
    )

    answers = []
    cnt_ = 1
    for question_dict in questions:
        print(f"Question {cnt_} out of {len(questions)}")
        cnt_ += 1
        question_number = list(question_dict.keys())[0]
        question = question_dict[question_number]
        time_start = time.time()
        output = qa(question)
        time_end = time.time()
        print("Took ", round(time_end - time_start, 2), " seconds")
        print("Answer: ", output['result'])
        print("Number of output tokens: " + str(llm.get_num_tokens(output['result'])))
        eval_dict["human_eval"] = get_human_eval_metric((i*3) + 1, (i+1)*3)
        answer_dict = {
            "question_number": question_number,
            "question": question,
            "query": output['query'],
            "answer": output['result'],
            "source_documents": [{
                "page_content": doc.page_content,
                "source_file": doc.metadata['source']
            } for doc in output['source_documents']
            ],
            "combined_documents": qa.combine_documents_chain._get_inputs(output['source_documents'])['context'],
            "time": round(time_end - time_start, 2),
            "evaluation": eval_dict,
        }
        answers.append(answer_dict)
        print("Number of input tokens: " + str(llm.get_num_tokens(answer_dict['combined_documents'])))
    #answer_file.write(json.dumps(answers, indent=4))

