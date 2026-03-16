import kagglehub
import os
import shutil

DATASETS = [
    "jenbati/ranjana-script-64",
    "shyboy123maharjan/ranjana-lipi",
]

def download_dataset(dataset):
    print(f"Downloading {dataset}...")
    cache_path = kagglehub.dataset_download(dataset)
    print(f"Cached at: {cache_path}")
    
    # copy to local data/ directory
    dataset_name = dataset.split("/")[-1]
    local_path = os.path.join("data", dataset_name)
    
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    shutil.copytree(cache_path, local_path)
    print(f"Copied to: {local_path}")
    return local_path

if __name__ == "__main__":
    for d in DATASETS:
        download_dataset(d)
    print("All datasets downloaded.")