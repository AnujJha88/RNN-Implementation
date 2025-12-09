import os
import tarfile
import urllib.request
import re
import random
import sys

# URL for the standard IMDB Sentiment Dataset
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
FILENAME = "aclImdb_v1.tar.gz"
DATA_FILE = "data.py"

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    """
    if total <= 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write('\n')

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<br />', ' ', text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Lowercase
    return text.lower().strip()

def download_hook(count, block_size, total_size):
    # Callback function for urlretrieve
    if total_size > 0:
        print_progress_bar(count * block_size, total_size, prefix='Downloading:', suffix='', length=40)

def download_and_extract():
    # 1. Download
    if not os.path.exists(FILENAME):
        print(f"Downloading corpus from {URL}...")
        try:
            urllib.request.urlretrieve(URL, FILENAME, reporthook=download_hook)
            print("Download complete.")
        except Exception as e:
            print(f"\nError downloading: {e}")
            return
    else:
        print(f"Found {FILENAME}, skipping download.")
    
    # 2. Extract
    if not os.path.exists("aclImdb"):
        print("Extracting files (calculating total files first)...")
        try:
            with tarfile.open(FILENAME, "r:gz") as tar:
                # getmembers() can be slow but necessary to know total count for progress bar
                members = tar.getmembers()
                total_files = len(members)
                
                for i, member in enumerate(members):
                    tar.extract(member)
                    if i % 100 == 0 or i == total_files - 1: # Update every 100 files to speed up
                        print_progress_bar(i + 1, total_files, prefix='Extracting: ', suffix='', length=40)
            print("Extraction complete.")
        except Exception as e:
            print(f"\nError extracting: {e}")
            return
    else:
        print("Directory 'aclImdb' already exists, skipping extraction.")

def build_data_file(num_samples=5000):
    print(f"\nProcessing data (Target: {num_samples} samples)...")
    
    data = {}
    total_processed = 0
    
    # Iterate through pos/neg folders
    label_types = ["pos", "neg"]
    
    # Pre-fetch file lists to enable progress bar for processing
    all_files = []
    for label_type in label_types:
        dir_path = os.path.join("aclImdb", "train", label_type)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            random.shuffle(files)
            # Assign label (1 for pos, 0 for neg)
            label_val = 1 if label_type == "pos" else 0
            # Take only what we need for this half
            limit = num_samples // 2
            all_files.extend([(os.path.join(dir_path, f), label_val) for f in files[:limit]])
    
    random.shuffle(all_files) # Shuffle mixed labels
    total_to_process = len(all_files)

    for i, (filepath, label_val) in enumerate(all_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = clean_text(content)
            
            # Filter logic
            if 10 < len(cleaned.split()) < 100:
                data[cleaned] = label_val
        
        # Update progress bar
        print_progress_bar(i + 1, total_to_process, prefix='Building Data:', suffix=f'({len(data)} kept)', length=40)
    
    print(f"\nWriting {len(data)} samples to {DATA_FILE}...")
    with open(DATA_FILE, "w", encoding='utf-8') as f:
        f.write("# Auto-generated huge dataset\n")
        f.write("train_data = {\n")
        for sentence, label in data.items():
            # Escape quotes
            safe_sentence = sentence.replace("'", "").replace('"', '')
            f.write(f'    "{safe_sentence}": {label},\n')
        f.write("}\n")
        f.write("\nval_data = train_data\n")

    print(f"Success! {DATA_FILE} created.")

if __name__ == "__main__":
    download_and_extract()
    # Note: Pure NumPy is slow. Start small to test your LSTM.
    build_data_file(num_samples=2000)