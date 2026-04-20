import os
import string
import nltk
import re
import math
import json
from typing import List
from nltk.corpus import stopwords
from src.models.schema import MinimalSource

class Indexer:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.list_minimal_sources: List[MinimalSource] = []
        self.all_chunks_tfs: List[dict[str, float]] = []
        self.global_df: dict[str, float] = {}
        self.chunk_vectors: List[dict[str, float]] = []
        self.chunks_processed: int = 0

    def normalize_docs(self, text: str) -> List[str]:
        # CAMBIO CLAVE: Sustituir puntuación por espacios, NO eliminarla
        punct = string.punctuation + "¡¿"
        table = str.maketrans(punct, ' ' * len(punct))
        text = text.translate(table).lower()
        return [w for w in text.split() if w not in self.stop_words and len(w) > 1]

    def normalize_code(self, text: str) -> List[str]:
        # CAMBIO CLAVE: Regex para mantener nombres de variables con guiones bajos
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
        return [w for w in tokens if w not in self.stop_words and len(w) > 1]

    def normalizer(self, text: str, is_code: bool = False) -> List[str]:
        return self.normalize_code(text) if is_code else self.normalize_docs(text)

    def build_index(self, ingester) -> None:
        from src.pipelines.ingester import DataIngester
        self.all_chunks_tfs = []
        self.list_minimal_sources = []
        self.global_df = {}
        self.chunks_processed = 0

        for path_file, content, suffix in ingester.search_files():
            is_code = suffix not in ['.md', '.txt', '.rst']
            for chunk_text, metadata in ingester.chuncker(path_file, content, suffix, max_chars=self.chunk_size):
                tokens = self.normalizer(chunk_text, is_code=is_code)
                if not tokens: continue
                
                chunk_tf: dict[str, float] = {}
                for token in tokens:
                    chunk_tf[token] = chunk_tf.get(token, 0) + 1
                    if token not in self.global_df:
                        self.global_df[token] = 0
                
                # Guardar presencia para el DF
                for token in set(tokens):
                    self.global_df[token] += 1
                
                self.all_chunks_tfs.append(chunk_tf)
                self.list_minimal_sources.append(metadata)
                self.chunks_processed += 1

    def calculate_idf(self) -> None:
        for word in self.global_df:
            # IDF con suavizado para evitar divisiones por cero
            self.global_df[word] = math.log(self.chunks_processed / (self.global_df[word]))

    def generate_vector(self) -> List[dict[str, float]]:
        self.calculate_idf()
        self.chunk_vectors = []
        for chunk_tf in self.all_chunks_tfs:
            vector = {w: tf * self.global_df[w] for w, tf in chunk_tf.items()}
            self.chunk_vectors.append(vector)
        return self.chunk_vectors

    def save_index(self) -> None:
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/index_vector.json', 'w') as f:
            json.dump(self.chunk_vectors, f)
        with open('data/processed/index_metadata.json', 'w') as f:
            json.dump([s.model_dump() for s in self.list_minimal_sources], f)
        with open('data/processed/index_global_df.json', 'w') as f:
            json.dump(self.global_df, f)
        with open('data/processed/index_config.json', 'w') as f:
            json.dump({"chunks_processed": self.chunks_processed}, f)