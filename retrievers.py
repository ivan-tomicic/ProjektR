"""
This script creates a database of information gathered from local text files.
"""

import faiss
from functools import partial
from transformers import AutoModel
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
)
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.utils import DistanceStrategy
from langchain.schema import Document, BaseRetriever
from langchain.schema.vectorstore import VectorStore
from langchain.schema.embeddings import Embeddings
from rerank_retriever import ReRankRetriever, BGE_ReRanker
from questions import get_random_question, get_question_tuples
from typing import Callable, Literal


def get_documents(folder: str):
    loader = DirectoryLoader(
        folder, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def embeddings_0():
    embedddings = OpenAIEmbeddings(openai_api_key='<KEY>')
    return embedddings


def embeddings_1():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )
    return embeddings


def embeddings_2():
    AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # langchain-ai/langchain/issues/6080
    embeddings = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-en",
        model_kwargs={"device": "cuda"},
    )
    return embeddings


def embeddings_3():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en", model_kwargs={"device": "cuda"}
    )
    return embeddings


def vectorstore_1(embeddings: Embeddings, index_name: Literal["l2", "ip"]):
    embedding_size = len(embeddings.embed_query("Example"))
    indexes = {
        "l2": lambda: faiss.IndexFlatL2(embedding_size),
        "ip": lambda: faiss.IndexFlatIP(embedding_size),
    }
    index = indexes[index_name]()
    return FAISS(
        embeddings,
        index,
        InMemoryDocstore(),
        {},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def rerank_1(base_retriever: BaseRetriever, top_n: int):
    reranker = BGE_ReRanker()
    retriever = ReRankRetriever(
        top_n=top_n, base_retriever=base_retriever, reranker=reranker
    )
    return retriever


def rerank_2(base_retriever: BaseRetriever, top_n: int):
    reranker = CohereRerank(top_n=top_n, cohere_api_key='<KEY>')
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )
    return retriever


def retriever_rerank(
    documents: list[Document],
    embeddings_func: Callable[[], Embeddings],
    vectorstore_func: Callable[[Embeddings], VectorStore],
    rerank_func: Callable[[BaseRetriever, int], BaseRetriever],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    number_of_retrievals: int = 10,
    top_number_of_retrievals: int = 3
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # splitter = SpacyTextSplitter(pipeline='en_core_web_lg', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #
    docs = splitter.split_documents(documents)
    #
    embeddings = embeddings_func()
    #
    db = vectorstore_func(embeddings)
    db.add_documents(docs)
    base_retriever = db.as_retriever(search_kwargs={"k": number_of_retrievals})
    #
    return rerank_func(base_retriever, top_number_of_retrievals)


def retriever_parent(
    documents: list[Document],
    embeddings_func: Callable[[], Embeddings],
    vectorstore_func: Callable[[Embeddings], VectorStore],
    *,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 200,
    number_of_retrievals: int = 5
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)
    #
    docstore = InMemoryStore()
    #
    embeddings = embeddings_func()
    #
    db = vectorstore_func(embeddings)
    #
    retriever = ParentDocumentRetriever(
        vectorstore=db,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=splitter,
        search_kwargs={"k": number_of_retrievals}
    )
    retriever.add_documents(documents)
    #
    return retriever


if __name__ == "__main__":
    documents = get_documents("./english_docs_od_mentora_txt/")

    # create and save the local database
    # FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=False) # Inner Product
    # FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE, normalize_L2=False) # Euclidean
    # FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True) # Cosine
    #
    # DistanceStrategy.MAX_INNER_PRODUCT i DistanceStrategy.EUCLIDEAN_DISTANCE su jedini implementirani tako
    #               da ne koristiti druge (npr. DistanceStrategy.COSINE). Ako se koriste druge distance, implementacija
    #               ce samo prebaciti na DistanceStrategy.EUCLIDEAN_DISTANCE
    #

    retriever = retriever_rerank(documents, embeddings_1, partial(vectorstore_1, index_name="l2"), rerank_1)

    queryId, queryQuestion = get_random_question("./test_results_new/questions.json")
    # for queryId, queryQuestion in get_question_tuples("./test_results_new/questions.json")

    texts = retriever.get_relevant_documents(queryQuestion)
    print("\n\n\n\n\n\n\n\n" + "-" * 400 + "\n\n\n\n\n\n\n\n")
    print(f"<{queryId}>: {queryQuestion}")
    for text in texts:
        try:
            print(text.page_content, end="\n\n\n|" + "-" * 50 + "|\n\n\n")
        except:
            print(
                "<Unable to print page content>", end="\n\n\n|" + "-" * 50 + "|\n\n\n"
            )
