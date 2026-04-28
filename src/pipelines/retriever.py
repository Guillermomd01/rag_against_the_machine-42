from typing import List

import bm25s

from src.models.schema import MinimalSource
from src.pipelines.indexer import Indexer


class Retriever:
    """Class responsible for retrieving relevant chunks based on a query."""

    def __init__(self) -> None:
        pass

    def retrieve(
        self,
        query_tokens_str: str,
        indexer: Indexer,
        top_k: int = 5,
        index_type: str = 'docs'
    ) -> List[MinimalSource]:
        """Retrieves the most relevant chunks from the specified index.

        Args:
            query_tokens_str: Query already tokenized and joined by spaces.
            indexer: Loaded Indexer instance.
            top_k: Number of results to retrieve.
            index_type: Either 'docs' or 'code'.

        Returns:
            List of MinimalSource with the top-k retrieved chunks.
        """
        if index_type == 'docs':
            retriever = indexer.docs_retriever
            metadata = indexer.docs_metadata
        elif index_type == 'code':
            retriever = indexer.code_retriever
            metadata = indexer.code_metadata
        else:
            return []

        if retriever is None or not metadata:
            return []

        query_tokens = bm25s.tokenize(
            [query_tokens_str],
            stopwords=None,
            stemmer=None,
            lower=False,
            token_pattern=r'\S+',
            allow_empty=False
        )
        try:
            results, scores = retriever.retrieve(query_tokens, k=top_k)
            top_indices = results[0]
            result_sources = [
                metadata[i] for i in top_indices if i < len(metadata)
            ]
            return result_sources
        except (ValueError, IndexError):
            return []

    def retrieve_combined(
        self,
        query_tokens_str: str,
        indexer: Indexer,
        top_k: int = 5
    ) -> List[MinimalSource]:
        """Retrieve from both indices and merge results."""
        docs_results = self.retrieve(
            query_tokens_str, indexer, top_k=top_k, index_type='docs')
        code_results = self.retrieve(
            query_tokens_str, indexer, top_k=top_k, index_type='code')
        # Simple interleaving: alternate docs and code results
        merged: List[MinimalSource] = []
        for d, c in zip(docs_results, code_results):
            merged.append(d)
            merged.append(c)
        # Append remaining from the longer list
        if len(docs_results) > len(code_results):
            merged.extend(docs_results[len(code_results):])
        else:
            merged.extend(code_results[len(docs_results):])
        return merged[:top_k]
