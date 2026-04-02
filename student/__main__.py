import uuid

import fire
from src.models.schema import MinimalSource
from src.pipelines.ingester import DataIngester
from src.pipelines.indexer import Indexer
from src.pipelines.retriever import Retriever
from src.models.schema import MinimalSearchResults, StudentSearchResults
import json

class RAGCLI:
    """Command Line Interface."""

    def index(self, max_chunk_size: int = 2000):
        """Indexa el repositorio[cite: 203]."""
        print(f"Iniciando indexación con chunks de {max_chunk_size}...")
        ingester = DataIngester(data_dir="data/raw")
        indexer = Indexer(chunk_size=max_chunk_size)
        indexer.build_index(ingester)
        indexer.generate_vector()
        indexer.save_index()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5):
        """Busca una única query."""
        print(f"Looking for {k} best results for: '{query}'")
        indexer = Indexer(chunk_size=2000)
        metadata = json.load(open('data/processed/index_metadata.json'))
        indexer.list_minimal_sources = [MinimalSource.model_validate(source) for source in metadata]
        indexer.global_df = json.load(open('data/processed/index_global_df.json'))
        indexer.chunk_vectors = json.load(open('data/processed/index_vector.json'))
        retriever = Retriever()
        results = retriever.retrieve(query, indexer, k)
        mini_search_results = [MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=results
        )]
        student_results = StudentSearchResults(search_results = mini_search_results, k = k)
        print(student_results.model_dump_json(indent=4))
    def search_dataset(self, dataset_path: str, k: int = 5, save_directory: str = "data/output"):
        """Procesa múltiples preguntas y exporta los resultados."""
        pass

    def answer(self, query: str, k: int = 10):
        """Responde a una pregunta usando contexto[cite: 206]."""
        pass

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        """Genera respuestas a partir de resultados de búsqueda[cite: 207]."""
        pass

    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10):
        """Evalúa los resultados contra el ground truth[cite: 208]."""
        pass

if __name__ == '__main__':
    fire.Fire(RAGCLI)