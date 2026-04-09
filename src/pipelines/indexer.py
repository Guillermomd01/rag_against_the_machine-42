import os
import string
import nltk
from typing import List
from nltk.corpus import stopwords
from src.pipelines.ingester import DataIngester
import math
import json
class Indexer:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.list_minimal_sources: List[DataIngester] = []
        self.local_tf: dict[str, int] = {}
        self.all_chunks_tfs: List[dict[str, int]] = []
        self.global_df: dict[str, int] = {}
        self.chunk_vectors: List[dict[str, float]] = []
        self.chuncks_processed: int = 0
    
    def normalizer(self, text: str) -> List[str]:
        """Normalizes the input text by removing punctuation,
        converting to lowercase, and removing stop words."""
        table = str.maketrans('', '', string.punctuation + "¡¿")
        text = text.translate(table)
        text_clean = text.lower().split()
        text_clean = [word for word in text_clean if word not in self.stop_words]
        return text_clean

    def build_index(self, ingester: DataIngester) -> None:
        """Builds the index by processing the ingested data."""
        for path, content, suffix in ingester.search_files():
            for chunk_text, metadata in ingester.chuncker(path, content, suffix):
                self.list_minimal_sources.append(metadata)
                normalized_text = self.normalizer(chunk_text)
                for word in normalized_text:
                    self.local_tf[word] = self.local_tf.get(word, 0) + 1
                for word in set(normalized_text):
                    if word not in self.global_df:
                        self.global_df[word] = 1
                    else:
                        self.global_df[word] += 1
                self.all_chunks_tfs.append(self.local_tf)
                self.local_tf = {}
                self.chuncks_processed += 1

    def calculate_idf(self) -> None:
        """Calculates the Inverse Document Frequency (IDF)
        for each word in the global document frequency."""
        for word in self.global_df.keys():
            df = self.global_df.get(word, 0)
            idf = math.log(self.chuncks_processed / (1 + df))
            self.global_df[word] = idf
    
    def generate_vector(self) -> List[dict[str, float]]:
        """Generates the vector representation for each chunk using TF-IDF."""
        self.calculate_idf()
        for chunk_tf in self.all_chunks_tfs:
            vector = {word: tf * self.global_df[word] for word, tf in chunk_tf.items()}
            self.chunk_vectors.append(vector)
        return self.chunk_vectors
    
    def save_index(self) -> None:
        """Saves the generated index to disk."""
        os.makedirs('data/processed', exist_ok=True)
        try:
            with open('data/processed/index_vector.json', 'w') as f:
                json.dump(self.chunk_vectors, f)
        except Exception as e:
            print(f"Error saving index vector: {e}")
        try:
            with open('data/processed/index_metadata.json', 'w') as f:
                dict_minimal_sources = [source.model_dump() for source in self.list_minimal_sources]
                json.dump(dict_minimal_sources, f)
        except Exception as e:
            print(f"Error saving index metadata: {e}")
        try:
            with open('data/processed/index_global_df.json', 'w') as f:
                json.dump(self.global_df, f)
        except Exception as e:
            print(f"Error saving index global df: {e}")