from pathlib import Path
from src.models.schema import MinimalSource

class DataIngester():
    """Class responsible for ingesting data from a specified directory."""
    def __init__(self, data_dir: str | Path):
        """Initializes the DataIngester with the given data directory."""
        self.data_dir = Path(data_dir)

    def search_files(self):
        """Searches for .py and .md files in the specified directory and reads their content."""
        for file in self.data_dir.rglob('*'):
            if file.is_file() and file.suffix in ['.py', '.md']:
                try:
                    with open(file, 'r',encoding='utf-8') as f:
                        path_file = str(file)
                        content = f.read()
                        yield (path_file, content, file.suffix)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
    def chuncker(self, path_file, content, suffix, max_chars=2000, overlap=200):
        start_ptr = 0
        total_len = len(content)

        while start_ptr < total_len:
            tentative_end = min(start_ptr + max_chars, total_len)
            
            if tentative_end < total_len:
                if suffix == ".md":
                    actual_end = content.rfind('\n\n', start_ptr, tentative_end)
                elif suffix == ".py":
                    
                    break_point1 = content.rfind('\ndef ', start_ptr, tentative_end)
                    break_point2 = content.rfind('\nclass ', start_ptr, tentative_end)
                    actual_end = max(break_point1, break_point2)
                    if actual_end == -1:
                        break
                
                # if rfind dont find nothing (return -1), we fallback to the tentative end
                if actual_end <= start_ptr:
                    actual_end = content.rfind('\n', start_ptr, tentative_end)
                if actual_end <= start_ptr:
                    actual_end = tentative_end
            else:
                actual_end = total_len

            chunk_text = content[start_ptr:actual_end]
            source_metadata = MinimalSource(
                file_path=path_file,
                first_character_index=start_ptr,
                last_character_index=actual_end
            )

            yield (chunk_text, source_metadata)
            if actual_end > start_ptr:
                start_ptr = actual_end - overlap
            if start_ptr < 0:
                start_ptr = 0