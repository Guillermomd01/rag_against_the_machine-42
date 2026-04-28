import re
import string
from pathlib import Path
from typing import Iterator, Tuple, List
from nltk.corpus import stopwords
import nltk

from src.models.schema import MinimalSource


class DataIngester():
    """Class responsible for ingesting data from a specified directory."""

    def __init__(self, data_dir: str | Path) -> None:
        """Initializes the DataIngester with the given data directory."""
        self.data_dir = Path(data_dir)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def search_files(self) -> Iterator[Tuple[str, str, str]]:
        """Searches for relevant files in the specified directory."""
        valid_extensions = {
            '.md', '.rst', '.txt', '.py', '.cpp', '.h', '.hpp',
            '.yaml', '.yml', '.toml', '.json', '.sh', '.in', '.inl',
            '.cu', '.cuh', '.c'
        }
        exclude_dirs = {
            '__pycache__', '.git', '.venv', 'node_modules',
            '.pytest_cache', 'build', 'dist', 'egg-info',
            'specs', 'third_party', '.circleci', '.github'
        }
        exclude_files = {
            'DCO', 'LICENSE', 'gitignore', 'gitkeep', 'gitattributes',
            'clang-format', 'yapfignore', 'dockerignore', 'helmignore',
            'nightly_torch', '.dockerignore', '.gitignore', '.yamllint'
        }
        for file in self.data_dir.rglob('*'):
            if file.is_file():
                if any(exclude in str(file) for exclude in exclude_dirs):
                    continue
                if file.name in exclude_files:
                    continue
                is_valid = file.suffix.lower() in valid_extensions
                is_readme = file.name == 'README'
                if is_valid or is_readme:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            path_file = str(file)
                            content = f.read()
                            if len(content) < 300:
                                continue
                            yield (path_file, content, file.suffix)
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")

    @staticmethod
    def _split_camel_case(token: str) -> List[str]:
        """Split camelCase and PascalCase tokens into separate words."""
        tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', token)
        tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tokens)
        return tokens.split()

    def normalize_docs(self, text: str) -> List[str]:
        """Normalize documentation text into tokens."""
        punct = string.punctuation + "¡¿"
        table = str.maketrans(punct, ' ' * len(punct))
        text = text.translate(table).lower()
        words = text.split()
        return [w for w in words
                if w not in self.stop_words and len(w) > 1]

    def normalize_code(self, text: str) -> List[str]:
        """Normalize code text into tokens, splitting identifiers."""
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
        result = []
        for token in tokens:
            token = token.lower()
            if token in self.stop_words or len(token) <= 1:
                continue
            parts = token.split('_')
            for part in parts:
                if len(part) > 1 and part not in self.stop_words:
                    result.append(part)
                camel_parts = self._split_camel_case(part)
                for cp in camel_parts:
                    cp_lower = cp.lower()
                    if len(cp_lower) > 1 and cp_lower not in self.stop_words:
                        result.append(cp_lower)
        return result

    def normalizer(self, text: str, is_code: bool = False) -> List[str]:
        """Normalize text based on whether it's code or documentation."""
        if is_code:
            return self.normalize_code(text)
        return self.normalize_docs(text)

    def chuncker(
        self, path_file: str, content: str, suffix: str,
            max_chars: int = 2000,
            overlap: int = 250) -> Iterator[Tuple[str, MinimalSource]]:
        """Splits the content into chunks of a specified maximum
        character length, with a specified overlap between chunks."""
        start_ptr = 0
        total_len = len(content)

        is_code = suffix in ['.py', '.cpp', '.h', '.hpp', '.sh',
                             '.cu', '.cuh', '.c']
        effective_max = 1500 if is_code else max_chars
        effective_overlap = 300 if is_code else overlap

        while start_ptr < total_len:
            tentative_end = min(start_ptr + effective_max, total_len)

            if tentative_end < total_len:
                actual_end = -1
                doc_suffixes = ['.md', '.rst', '.txt', '.yaml', '.yml',
                                '.toml', '.json']
                code_suffixes = ['.py', '.cpp', '.h', '.hpp', '.c',
                                 '.cu', '.cuh']
                if suffix in doc_suffixes:
                    actual_end = content.rfind('\n\n', start_ptr,
                                               tentative_end)
                    if actual_end <= start_ptr:
                        actual_end = content.rfind('\n', start_ptr,
                                                   tentative_end)
                elif suffix in code_suffixes:
                    break_points = [
                        content.rfind('\ndef ', start_ptr, tentative_end),
                        content.rfind('\nclass ', start_ptr, tentative_end),
                        content.rfind('\nasync def ', start_ptr,
                                      tentative_end),
                        content.rfind('\n    def ', start_ptr,
                                      tentative_end),
                        content.rfind('\n# ', start_ptr, tentative_end),
                    ]
                    actual_end = max(break_points)
                elif suffix == '.sh':
                    actual_end = content.rfind('\nfunction ', start_ptr,
                                               tentative_end)
                    if actual_end <= start_ptr:
                        actual_end = content.rfind('\n', start_ptr,
                                                   tentative_end)
                else:
                    actual_end = content.rfind('\n', start_ptr,
                                               tentative_end)

                if actual_end <= start_ptr:
                    actual_end = tentative_end
            else:
                actual_end = total_len

            chunk_text = content[start_ptr:actual_end]
            if len(chunk_text) < 100 and actual_end < total_len:
                start_ptr = actual_end
                continue

            source_metadata = MinimalSource(
                file_path=path_file,
                first_character_index=start_ptr,
                last_character_index=actual_end
            )

            yield (chunk_text, source_metadata)

            next_start = actual_end - effective_overlap
            if next_start <= start_ptr:
                next_start = actual_end
            start_ptr = next_start
