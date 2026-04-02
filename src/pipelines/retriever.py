
from typing import List
from src.models.schema import MinimalSource
from src.pipelines.indexer import Indexer


class Retriever():
    def __init__(self):
        pass
    
    def retrieve(self, query: str, indexer:Indexer, top_k: int = 5) -> list[MinimalSource]:
        """Retrieves the most relevant chunks based on the query."""
        scores = self._calculate_similarities(query, indexer)
        top_scores = scores[:top_k]
        results = [indexer.list_minimal_sources[index] for index, _ in top_scores]
        return results

    def _vectorize_query(self, query: str, indexer: Indexer) -> dict[str, float]:
        """Vectorizes the query using the same TF-IDF approach as the index."""
        query_normalized = indexer.normalizer(query)
        query_tf = {}
        for word in query_normalized:
            query_tf[word] = query_tf.get(word, 0) + 1
        query_vector = {word: tf * indexer.global_df.get(word, 0) for word, tf in query_tf.items()}
        return query_vector        
        
    def _calculate_similarities(self, query: str, indexer: Indexer) -> List[tuple[int, float]]:
        """Retrieves the most relevant chunks based on the query."""
        query_vector = self._vectorize_query(query, indexer)
        scores = []
        for index, vector in enumerate(indexer.chunk_vectors):
            score_vector = 0
            for word, weight in query_vector.items():
                if word in vector:
                    score_vector += weight * vector[word]
            scores.append((index, score_vector))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores