import uuid
from tqdm import tqdm
import os
from pathlib import Path
import fire
from src.models.schema import MinimalSource
from src.pipelines.ingester import DataIngester
from src.pipelines.indexer import Indexer
from src.pipelines.retriever import Retriever
from src.pipelines.generator import AnswerGenerator
from src.models.schema import MinimalSearchResults, StudentSearchResults
from src.models.schema import MinimalAnswer, StudentAnswerResults
import json

class RAGCLI:
    """Command Line Interface."""

    def index(self, max_chunk_size: int = 2000):
        """Indexa el repositorio."""
        print(f"Iniciando indexación con chunks de {max_chunk_size}...")
        ingester = DataIngester(data_dir="data/raw")
        indexer = Indexer(chunk_size=max_chunk_size)
        indexer.build_index(ingester)
        indexer.generate_vector()
        indexer.save_index()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5, save_path: str = None):
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
        json_output = student_results.model_dump_json(indent=4)
        
        if save_path:
            save_dir = "data/output/single_queries/search"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, save_path)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n¡Result saved: {path}!")
        else:
            print("\n" + json_output)

    def search_dataset(self, dataset_path: str, k: int = 5, save_directory: str = "data/output/datasets/search"):
        """Procesa múltiples preguntas y exporta los resultados."""
        print(f"Processing dataset: {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        indexer = Indexer(chunk_size=2000)
        metadata = json.load(open('data/processed/index_metadata.json'))
        indexer.list_minimal_sources = [MinimalSource.model_validate(source) for source in metadata]
        indexer.global_df = json.load(open('data/processed/index_global_df.json'))
        indexer.chunk_vectors = json.load(open('data/processed/index_vector.json'))
        
        retriever = Retriever()
        all_mini_results = []

        for item in tqdm(dataset.get('rag_questions', []), desc="Processing questions"):
            question_id = item['question_id']
            query = item['question']

            results = retriever.retrieve(query, indexer, k)
            mini_result = MinimalSearchResults(
                question_id=question_id,
                question=query,
                retrieved_sources=results
            )
            all_mini_results.append(mini_result)            
        student_results = StudentSearchResults(search_results=all_mini_results, k=k)
        
        os.makedirs(save_directory, exist_ok=True)
        filename = "results_" + Path(dataset_path).name
        save_path = os.path.join(save_directory, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(student_results.model_dump_json(indent=4))
            
        print(f"Results: {save_path}")
    
    def answer(self, query: str, k: int = 10, save_path: str = None):
        """Responde a una pregunta usando contexto."""
        print(f"Looking for context for: '{query}'...")
        
        indexer = Indexer(chunk_size=2000)
        metadata = json.load(open('data/processed/index_metadata.json'))
        indexer.list_minimal_sources = [MinimalSource.model_validate(source) for source in metadata]
        indexer.global_df = json.load(open('data/processed/index_global_df.json'))
        indexer.chunk_vectors = json.load(open('data/processed/index_vector.json'))
        
        retriever = Retriever()
        retrieved_sources = retriever.retrieve(query, indexer, k)
        
        context_texts = []
        for source in retrieved_sources:
            with open(source.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunk_text = content[source.first_character_index : source.last_character_index]
                context_texts.append(chunk_text)

        with tqdm(total=1, desc="Downloading model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            generator = AnswerGenerator()
            pbar.update(1)
            
        print("\nCreating answer...")
        answer_text = generator.generate_answer(query, context_texts)
        
        mini_answer = MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=retrieved_sources,
            answer=answer_text
        )
        
        student_results = StudentAnswerResults(answer_results=[mini_answer], k=k)        
        json_output = student_results.model_dump_json(indent=4)
        
        if save_path:
            save_dir = "data/output/single_queries/answer"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, save_path)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n¡Result saved: {path}!")
        else:
            print("\n" + json_output)
        
    def answer_dataset(self, student_search_results_path: str, save_directory: str = "data/output/datasets/answer"):
        """Genera respuestas a partir de resultados de búsqueda previos."""
        print(f"Generating answers for results in: {student_search_results_path}...")
        
        with open(student_search_results_path, 'r', encoding='utf-8') as f:
            search_data = json.load(f)
            
        results_list = search_data.get('search_results', [])
        k = search_data.get('k', 10)
        
        with tqdm(total=1, desc="Loading Qwen", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            generator = AnswerGenerator()
            pbar.update(1)

        all_answers = []
        for item in tqdm(results_list, desc="Generating answers"):
            question_id = item['question_id']
            query = item['question']
            sources_dicts = item['retrieved_sources']
            
            context_texts = []
            retrieved_sources_objs = []
            
            for source_dict in sources_dicts:
                source_obj = MinimalSource(**source_dict)
                retrieved_sources_objs.append(source_obj)
                
                with open(source_obj.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunk_text = content[source_obj.first_character_index : source_obj.last_character_index]
                    context_texts.append(chunk_text)
                    
            answer_text = generator.generate_answer(query, context_texts)
            
            mini_answer = MinimalAnswer(
                question_id=question_id,
                question=query,
                retrieved_sources=retrieved_sources_objs,
                answer=answer_text
            )
            all_answers.append(mini_answer)
            
        student_answers = StudentAnswerResults(answer_results=all_answers, k=k)
        
        os.makedirs(save_directory, exist_ok=True)
        filename = Path(student_search_results_path).name 
        save_path = os.path.join(save_directory, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(student_answers.model_dump_json(indent=4))
            
        print(f"\n¡Answers generated: {save_path}")
        
    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10):
        """Evalúa los resultados contra el ground truth."""
        pass

if __name__ == '__main__':
    fire.Fire(RAGCLI)