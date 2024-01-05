import torch
import itertools
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import Any, Callable, Iterable, List


class BGE_ReRanker:
    def __init__(self, *, tokenizer_kwargs: dict[str, Any] = {}, model_kwargs: dict[str, Any] = {}):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large", **tokenizer_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-large",
            **model_kwargs
        )
        self.model.eval()

    def __call__(self, pairs: list[tuple[str, str]]) -> Iterable[int]:
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = self.model(**inputs, return_dict=True).logits.squeeze()
            return scores.argsort(descending=True)


class ReRankRetriever(BaseRetriever):
    top_n: int

    base_retriever: BaseRetriever

    reranker: Callable[[list[tuple[str, str]]], Iterable[int]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        sub_docs = self.base_retriever._get_relevant_documents(
            query, run_manager=run_manager
        )

        pairs = list(
            zip(itertools.repeat(query), map(lambda d: d.page_content, sub_docs))
        )

        return [sub_docs[i] for i in self.reranker(pairs)][:self.top_n]
