import uuid
from tqdm import tqdm
import os
from pathlib import Path
import fire
import json

from src.models.schema import MinimalSource
from src.pipelines.ingester import DataIngester
from src.pipelines.indexer import Indexer
from src.pipelines.retriever import Retriever
from src.pipelines.generator import AnswerGenerator
from src.pipelines.evaluator import Evaluator
from src.models.schema import MinimalSearchResults, StudentSearchResults
from src.models.schema import MinimalAnswer, StudentSearchResultsAndAnswer


def _detect_index_type(dataset_path: str) -> str:
    """Detect whether a dataset is docs or code based on its filename."""
    lower_path = dataset_path.lower()
    if 'docs' in lower_path:
        return 'docs'
    elif 'code' in lower_path:
        return 'code'
    return 'docs'


class RAGCLI:
    """Command Line Interface for the RAG system."""

    def __init__(self) -> None:
        """Initialize RAGCLI with lazy loading of models and indices."""
        self._generator: AnswerGenerator | None = None
        self._indexer: Indexer | None = None
        self._ingester: DataIngester | None = None
        self._retriever: Retriever | None = None
        self._file_cache: dict[str, str] = {}

    def _get_generator(self) -> AnswerGenerator:
        """Lazy load the AnswerGenerator (singleton pattern)."""
        if self._generator is None:
            with tqdm(total=1, desc="Loading model",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
                self._generator = AnswerGenerator()
                pbar.update(1)
        return self._generator

    def _get_indexer(self) -> Indexer:
        """Lazy load the Indexer with both indices."""
        if self._indexer is None:
            self._indexer = Indexer(chunk_size=2000)
            if not self._indexer.load_index('docs'):
                print("Warning: Docs index not found. Run 'index' first.")
            if not self._indexer.load_index('code'):
                print("Warning: Code index not found.")
        return self._indexer

    def _get_ingester(self) -> DataIngester:
        """Lazy load the DataIngester."""
        if self._ingester is None:
            self._ingester = DataIngester(data_dir="data/raw")
        return self._ingester

    def _get_retriever(self) -> Retriever:
        """Lazy load the Retriever."""
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the documents in data/raw and save
        the indices in data/processed."""
        print(f"Starting indexing with chunks of {max_chunk_size}...")
        print("Ingesting and chunking data...")
        ingester = DataIngester(data_dir="data/raw")
        print("Initializing indexer...")
        indexer = Indexer(chunk_size=max_chunk_size)
        print("Building index...")
        indexer.build_index(ingester)
        print("Generating vectors...")
        indexer.generate_vectors()
        print("Saving index...")
        indexer.save_index()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(
        self, query: str, k: int = 5,
            index_type: str = 'both',
            save_path: str | None = None) -> None:
        """Search for relevant chunks given a query
        and return the top k results."""
        print(f"Looking for {k} best results for: '{query}'")
        indexer = Indexer(chunk_size=2000)
        if not indexer.load_index('docs'):
            print("Error: Docs index not found. Run 'index' first.")
            return
        if not indexer.load_index('code'):
            print("Warning: Code index not found.")

        ingester = DataIngester(data_dir="data/raw")
        retriever = Retriever()

        if index_type == 'docs':
            tokens = ingester.normalizer(query, is_code=False)
            query_str = " ".join(tokens)
            results = retriever.retrieve(
                query_str, indexer, k, 'docs')
        elif index_type == 'code':
            tokens = ingester.normalizer(query, is_code=True)
            query_str = " ".join(tokens)
            results = retriever.retrieve(
                query_str, indexer, k, 'code')
        else:
            tokens_doc = ingester.normalizer(query, is_code=False)
            tokens_code = ingester.normalizer(query, is_code=True)
            query_doc = " ".join(tokens_doc)
            query_code = " ".join(tokens_code)
            results = retriever.retrieve_combined(
                query_doc + " " + query_code, indexer, k)

        mini_search_results = [MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question_str=query,
            retrieved_sources=results
        )]
        student_results = StudentSearchResults(
            search_results=mini_search_results, k=k)
        json_output = student_results.model_dump_json(indent=4)

        if save_path:
            save_dir = "data/output/single_queries/search"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, save_path)
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                print(f"\nResult saved: {path}!")
            except Exception as e:
                print(f"Error saving result: {e}")
        else:
            print("\n" + json_output)

    def search_dataset(
        self, dataset_path: str, k: int = 10,
            save_directory: str = "data/output/search") -> None:
        """Process a dataset of questions and
        save the search results."""
        print(f"Processing dataset: {dataset_path}...")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return

        idx_type = _detect_index_type(dataset_path)
        print(f"Detected index type: {idx_type}")

        indexer = Indexer(chunk_size=2000)
        if not indexer.load_index(idx_type):
            print(f"Error: {idx_type} index not found. Run 'index' first.")
            return

        ingester = DataIngester(data_dir="data/raw")
        retriever = Retriever()
        all_mini_results = []

        for item in tqdm(dataset.get('rag_questions', []),
                         desc="Processing questions"):
            question_id = item['question_id']
            query = item['question']

            tokens = ingester.normalizer(
                query, is_code=(idx_type == 'code'))
            query_str = " ".join(tokens)
            results = retriever.retrieve(
                query_str, indexer, k, idx_type)

            mini_result = MinimalSearchResults(
                question_id=question_id,
                question_str=query,
                retrieved_sources=results
            )
            all_mini_results.append(mini_result)

        student_results = StudentSearchResults(
            search_results=all_mini_results, k=k)

        os.makedirs(save_directory, exist_ok=True)
        filename = "results_" + Path(dataset_path).name
        save_path = os.path.join(save_directory, filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(student_results.model_dump_json(indent=4))

        print(f"Results: {save_path}")

    def answer(
        self, query: str, k: int = 10,
            save_path: str | None = None) -> None:
        """Answer one or more questions by retrieving
        relevant chunks and generating answers.
        For multiple questions, pass them space-separated
        or call answer multiple times."""
        # Support multiple queries separated by "|"
        queries = [q.strip() for q in query.split("|") if q.strip()]

        if not queries:
            print("Error: No valid queries provided.")
            return

        indexer = self._get_indexer()
        if indexer.docs_retriever is None and indexer.code_retriever is None:
            print("Error: No indices loaded. Run 'index' first.")
            return

        ingester = self._get_ingester()
        retriever = self._get_retriever()
        generator = self._get_generator()

        all_answers = []

        for idx, single_query in enumerate(queries, 1):
            display_query = (single_query[:57] + "..."
                             if len(single_query) > 60
                             else single_query)
            print(f"\n[{idx}/{len(queries)}] Looking for context for: "
                  f"'{display_query}'")

            tokens_doc = ingester.normalizer(single_query, is_code=False)
            tokens_code = ingester.normalizer(single_query, is_code=True)
            query_doc = " ".join(tokens_doc)
            query_code = " ".join(tokens_code)
            retrieved_sources = retriever.retrieve_combined(
                query_doc + " " + query_code, indexer, k)

            context_texts = []
            for source in retrieved_sources:
                file_path = source.file_path
                if file_path not in self._file_cache:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self._file_cache[file_path] = f.read()
                    except FileNotFoundError as e:
                        print(f"Error loading file: {e}")
                        return

                content = self._file_cache[file_path]
                chunk_text = content[
                    source.first_character_index:
                        source.last_character_index]
                context_texts.append(chunk_text)

            print("Creating answer...")
            answer_text = generator.generate_answer(
                single_query, context_texts)

            mini_answer = MinimalAnswer(
                question_id=str(uuid.uuid4()),
                question_str=single_query,
                retrieved_sources=retrieved_sources,
                answer=answer_text
            )
            all_answers.append(mini_answer)

        student_results = StudentSearchResultsAndAnswer(
            search_results=all_answers, k=k)
        json_output = student_results.model_dump_json(indent=4)

        if save_path:
            save_dir = "data/output/single_queries/answer"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, save_path)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\nResult saved: {path}!")
        else:
            print("\n" + json_output)

    def answer_dataset(
        self, student_search_results_path: str,
            save_directory: str = "data/output/datasets/answer") -> None:
        """Generate answers for a dataset of search results."""
        print(f"Generating answers for results in:"
              f" {student_search_results_path}...")
        try:
            with open(student_search_results_path, 'r',
                      encoding='utf-8') as f:
                search_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error loading search results: {e}")
            return
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return

        results_list = search_data.get('search_results', [])
        k = search_data.get('k', 10)

        generator = self._get_generator()

        all_answers = []

        for item in tqdm(results_list, desc="Generating answers"):
            question_id = item['question_id']
            query = item.get('question_str', item.get('question', ''))
            sources_dicts = item['retrieved_sources']

            context_texts = []
            retrieved_sources_objs = []
            for source_dict in sources_dicts[:3]:
                source_obj = MinimalSource(**source_dict)
                retrieved_sources_objs.append(source_obj)

                file_path = source_obj.file_path
                if file_path not in self._file_cache:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self._file_cache[file_path] = f.read()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                content = self._file_cache[file_path]
                chunk_text = content[
                    source_obj.first_character_index:
                        source_obj.last_character_index]
                context_texts.append(chunk_text)

            answer_text = generator.generate_answer(query, context_texts)

            mini_answer = MinimalAnswer(
                question_id=question_id,
                question_str=query,
                retrieved_sources=[MinimalSource(**s)
                                   for s in sources_dicts],
                answer=answer_text
            )
            all_answers.append(mini_answer)

        student_answers = StudentSearchResultsAndAnswer(
            search_results=all_answers, k=k)

        os.makedirs(save_directory, exist_ok=True)
        filename = Path(student_search_results_path).name
        save_path = os.path.join(save_directory, filename)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(student_answers.model_dump_json(indent=4))
            print(f"\nAnswers generated: {save_path}")
        except Exception as e:
            print(f"Error saving answers: {e}")
            return

    def evaluate(
        self, student_answer_path: str,
            dataset_path: str, k: int = 5,
            save_directory: str = "data/output/evaluation") -> None:
        """Evaluate the search dataset results against the ground truth
        and save the metrics."""
        print(f"Evaluating:\n"
              f"- Predictions: {student_answer_path}\n -"
              f"Ground Truth: {dataset_path}")
        evaluator = Evaluator()

        metrics_dict = evaluator.evaluate(
            student_answer_path, dataset_path, k)
        if not metrics_dict:
            print("\n[Error] Evaluation failed:"
                  "No valid metrics found.")
            return
        os.makedirs(save_directory, exist_ok=True)
        filename = "metrics_" + Path(student_answer_path).name
        save_path = os.path.join(save_directory, filename)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
            print(f"\nMetrics saved: {save_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
            return


if __name__ == '__main__':
    _script_dir = Path(__file__).resolve().parent
    if _script_dir.name == 'student':
        os.chdir(_script_dir.parent)
    elif _script_dir.name == 'src':
        os.chdir(_script_dir.parent)

    fire.Fire(RAGCLI)
