# utils/upload.py
from huggingface_hub import HfApi

def upload_to_huggingface(file_path, repo_id, filename):
    api = HfApi()
    api.upload_file(path_or_fileobj=file_path, path_in_repo=filename, repo_id=repo_id)
    print(f"Uploaded {filename} to {repo_id}")