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
        "model": "TheBloke/Llama-2-7B-GGUF",
        "model_file": "llama-2-7b.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/Llama-2-7b-Chat-GGUF",
        "model_file": "llama-2-7b-chat.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/Mistral-7B-Claude-Chat-GGUF",
        "model_file": "Mistral-7B-claude-chat.q5_k_m.gguf"
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
        "model": "TheBloke/Mistral-7B-v0.1-GGUF",
        "model_file": "mistral-7b-v0.1.Q5_K_M.gguf"
    },
    {
        "model": "Undi95/Mistral-ClaudeLimaRP-v3-7B-GGUF",
        "model_file": "Mistral-ClaudeLimaRP-v3-7B.q5_k_m.gguf"
    },
    {
        "model": "TheBloke/zephyr-7B-alpha-GGUF",
        "model_file": "zephyr-7b-alpha.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/zephyr-7B-beta-GGUF",
        "model_file": "zephyr-7b-beta.Q5_K_M.gguf"
    },
    {
        "model": "TheBloke/zephyr-7B-beta-GGUF",
        "model_file": "zephyr-7b-beta.Q5_K_S.gguf"
    }
]


template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


llm = CTransformers(
    model="TheBloke/zephyr-7B-alpha-GGUF",
    model_file="zephyr-7b-alpha.Q5_K_M.gguf",
    model_type="llama",
    config={'max_new_tokens': 256, 'temperature': 0.01}
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'})

db = FAISS.load_local("faiss", embeddings)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})


qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs={'prompt': prompt}
                                 )


with open("test_results_old/questions_for_testing_old.json", "r", encoding='utf-8') as f:
    questions = json.load(f)
cnt_ = 1

for question in questions.values():
    time_start = time.time()
    output = qa.run(question)
    time_end = time.time()
    print(f"#Question {cnt_}#:\n")
    print(question)

    print("\n\n#Answer#:\n")
    print(output)
    print(f"\n#Time taken#: {(time_end - time_start):.1f}")
    print()
    print("-"*30)
    print()
    cnt_ += 1
