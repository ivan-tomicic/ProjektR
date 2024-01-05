from functools import partial
from langchain.chains import RetrievalQA
from langchain.llms.ctransformers import CTransformers
from langchain.prompts import PromptTemplate
from retrievers import (
    get_documents,
    embeddings_0,
    embeddings_1,
    embeddings_2,
    embeddings_3,
    vectorstore_1,
    rerank_1,
    rerank_2,
    retriever_rerank,
    retriever_parent,
)
from questions import get_random_question, get_question_tuples
from typing import Literal


def get_retriever(retriever_type: Literal["parent", "rerank"]):
    documents = get_documents("./english_docs_od_mentora_txt/")
    if retriever_type == "parent":
        return retriever_parent(
            documents,
            [embeddings_0, embeddings_1, embeddings_2, embeddings_3][1],
            partial(vectorstore_1, index_name="l2"),  # ili "ip", nisam vidio razliku izmedu njih
            parent_chunk_size=2000,  # treba biti veci od child_chunk_size
            child_chunk_size=200,
            number_of_retrievals=5,
        )
    elif retriever_type == "rerank":
        return retriever_rerank(
            documents,
            [embeddings_0, embeddings_1, embeddings_2, embeddings_3][1],
            partial(vectorstore_1, index_name="l2"),  # ili "ip", nisam vidio razliku izmedu njih
            [rerank_1, rerank_2][0],
            chunk_size=1000,
            chunk_overlap=200,
            number_of_retrievals=10, # treba biti veci od top_number_of_retrievals
            top_number_of_retrievals=3,
        )
    else:
        raise ValueError("Invalid retriever type")


retriever = get_retriever("parent")

llm = CTransformers(
    model="TheBloke/Mistral-7B-OpenOrca-GGUF",
    model_file="mistral-7b-openorca.Q4_0.gguf",
    config={"max_new_tokens": 256, "temperature": 0.01, "context_length": 2048},
)

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # ili "refine"
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

queryId, queryQuestion = get_random_question("./final_grading/questions.json")
# ili for queryId, queryQuestion in get_question_tuples("./test_results_new/questions.json")

result = qa(queryQuestion)

print("QueryID: ", queryId, end="\n\n")
print("Query: ", result["query"], end="\n\n")
print("Source documents: ", result["source_documents"], end="\n\n")
print(
    "Combined documents: ",
    qa.combine_documents_chain._get_inputs(result["source_documents"])["context"],
    end="\n\n",
)
print("Answer: ", result["result"], end="\n\n")
