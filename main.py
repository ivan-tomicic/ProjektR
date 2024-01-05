import json
import time
import torch

from langchain.chains import RetrievalQA
from langchain.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from retrievers import *
from configurations import *


llm = CTransformers(
    model="TheBloke/Mistral-7B-OpenOrca-GGUF",
    model_file="mistral-7b-openorca.Q4_0.gguf",
    config={'max_new_tokens': 256, 'repetition_penalty': 1.1, 'context_length': 4000, 'temperature': 0.01},
)

documents = get_documents("./english_docs_od_mentora_txt/")

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

with open("final_test/questions.json", "r", encoding='utf-8') as f:
    questions = json.load(f)


eval_dict = {
    "bleu_score": -100_000,
    "rouge_score": {
        "precision": -100_000,
        "recall": -100_000,
        "f1": -100_000
    },
    "diversity": -100_000,
    "human_eval": {
        "expression_and_logic": "_",
        "accuracy_and_relevance": "_",
        "overall_quality_and_engagement": "_",
    },
}
i = 0
for config_name, config in configurations.items():
    i += 1
    print(f"Processing configuration: {config_name}")
    if i < 2:
        continue

    retriever = process_configuration(config, documents)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # ili "refine"
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    results = {}
    answers = []
    cnt_ = 1
    answer_file = open(f"final_test/answers/{config_name}.json", "w+", encoding='utf-8')
    results['config'] = convert_functions_to_names(config)
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
        answer_dict = {
            "question_number": question_number,
            "question": question,
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

    results['answers'] = answers
    answer_file.write(json.dumps(results, indent=4))
