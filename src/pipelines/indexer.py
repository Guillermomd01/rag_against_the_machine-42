import os
import re
import json
from pathlib import Path
from typing import Any, List

import bm25s

from src.models.schema import MinimalSource
from src.pipelines.ingester import DataIngester


class Indexer:
    """Builds and manages separate BM25 indices for docs and code."""

    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.docs_metadata: List[MinimalSource] = []
        self.code_metadata: List[MinimalSource] = []
        self.docs_corpus: List[str] = []
        self.code_corpus: List[str] = []
        self.docs_retriever: Any = None
        self.code_retriever: Any = None

    @staticmethod
    def _is_doc_file(suffix: str) -> bool:
        return suffix.lower() in {
            '.md', '.rst', '.txt', '.yaml', '.yml', '.toml', '.json'
        }

    @staticmethod
    def _is_code_file(suffix: str) -> bool:
        return suffix.lower() in {
            '.py', '.cpp', '.h', '.hpp', '.sh', '.cu', '.cuh', '.c',
            '.inl', '.in'
        }

    @staticmethod
    def _extract_path_tokens(path_file: str) -> List[str]:
        """Extract tokens from directory names and file stem."""
        path = Path(path_file)
        tokens = []
        for part in path.parts:
            name = Path(part).stem
            parts = re.split(r'[_\-\.]', name)
            for p in parts:
                p_lower = p.lower()
                if len(p_lower) > 1:
                    tokens.append(p_lower)
        return tokens

    def build_index(self, ingester: DataIngester) -> None:
        """Build separate corpora for docs and code."""
        self.docs_metadata = []
        self.code_metadata = []
        self.docs_corpus = []
        self.code_corpus = []

        for path_file, content, suffix in ingester.search_files():
            is_doc = self._is_doc_file(suffix)
            is_code = self._is_code_file(suffix)
            if not is_doc and not is_code:
                continue

            path_tokens = self._extract_path_tokens(path_file)
            kwargs = {'max_chars': self.chunk_size}
            for chunk_text, metadata in ingester.chuncker(
                    path_file, content, suffix, **kwargs):
                tokens = ingester.normalizer(chunk_text, is_code=is_code)
                if not tokens:
                    continue

                if is_code:
                    tokens = (path_tokens * 3) + tokens

                tokenized_text = " ".join(tokens)

                if is_doc:
                    self.docs_corpus.append(tokenized_text)
                    self.docs_metadata.append(metadata)
                elif is_code:
                    self.code_corpus.append(tokenized_text)
                    self.code_metadata.append(metadata)

        print(f"Docs chunks: {len(self.docs_corpus)}")
        print(f"Code chunks: {len(self.code_corpus)}")

    def _build_bm25(self, corpus: List[str]) -> Any:
        """Build a BM25 index from a tokenized corpus."""
        if not corpus:
            return None
        corpus_tokens = bm25s.tokenize(
            corpus,
            stopwords=None,
            stemmer=None,
            lower=False,
            token_pattern=r'\S+',
            allow_empty=False
        )
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        return retriever

    def generate_vectors(self) -> None:
        """Generate BM25 indices for docs and code."""
        print("Building docs BM25 index...")
        self.docs_retriever = self._build_bm25(self.docs_corpus)
        print("Building code BM25 index...")
        self.code_retriever = self._build_bm25(self.code_corpus)

    def save_index(self) -> None:
        """Save indices and metadata to disk."""
        base_path = 'data/processed'
        os.makedirs(base_path, exist_ok=True)

        docs_path = os.path.join(base_path, 'docs_index')
        os.makedirs(docs_path, exist_ok=True)
        if self.docs_retriever is not None:
            self.docs_retriever.save(docs_path)
        with open(os.path.join(docs_path, 'metadata.json'),
                  'w', encoding='utf-8') as f:
            json.dump([m.model_dump() for m in self.docs_metadata], f)

        code_path = os.path.join(base_path, 'code_index')
        os.makedirs(code_path, exist_ok=True)
        if self.code_retriever is not None:
            self.code_retriever.save(code_path)
        with open(os.path.join(code_path, 'metadata.json'),
                  'w', encoding='utf-8') as f:
            json.dump([m.model_dump() for m in self.code_metadata], f)

    def load_index(self, index_type: str) -> bool:
        """Load a BM25 index and its metadata."""
        base_path = 'data/processed'
        if index_type == 'docs':
            idx_path = os.path.join(base_path, 'docs_index')
            meta_file = os.path.join(idx_path, 'metadata.json')
            if not os.path.exists(meta_file):
                return False
            with open(meta_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                self.docs_metadata = [
                    MinimalSource.model_validate(m) for m in raw
                ]
            self.docs_retriever = bm25s.BM25.load(idx_path)
            return True
        elif index_type == 'code':
            idx_path = os.path.join(base_path, 'code_index')
            meta_file = os.path.join(idx_path, 'metadata.json')
            if not os.path.exists(meta_file):
                return False
            with open(meta_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                self.code_metadata = [
                    MinimalSource.model_validate(m) for m in raw
                ]
            self.code_retriever = bm25s.BM25.load(idx_path)
            return True
        return False
