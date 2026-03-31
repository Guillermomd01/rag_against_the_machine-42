import string
import nltk
from typing import List
from nltk.corpus import stopwords
from src.pipelines.ingester import DataIngester
import math
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
        for path, content, suffix in ingester.search_files():
            for chunk_text, metadata in ingester.chuncker(path, content, suffix):
                self.list_minimal_sources.append(metadata)
                normalized_text = self.normalizer(chunk_text)
                for word in normalized_text:
                    self.local_tf[word] = self.local_tf.get(word, 0) + 1
                    
                self.all_chunks_tfs.append(self.local_tf)
                self.local_tf = {}
                self.chuncks_processed += 1
                
                
    #def process_chuncks(self, chunks: DataIngester) -> None:
    #    text, metadata = chunks.chuncker()
    #    normalized_text = self.normalizer(text)
    #    for word in normalized_text:
    #        self.dict_tf[word] = self.dict_tf.get(word, 0) + 1
    #    self.list_minimal_sources.append(metadata)
    #    self.chuncks_processed += 1

    def calculate_idf(self) -> None:
        for word in self.dict_tf.keys():
            df = self.dict_df.get(word, 0)
            idf = math.log(self.chuncks_processed / (1 + df))
            self.dict_df[word] = idf
    
    def generate_vector(self) -> dict[str, float]:
        self.calculate_idf()
        vector = {word: tf * self.dict_df[word] for word, tf in self.dict_tf.items()}
        return vector