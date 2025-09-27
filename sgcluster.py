import os
import json
from collections import Counter

folder_path = "./data_sg"
output_dir = f"{folder_path}/keys"
json_files = {
    "sg1": f"{folder_path}/sg1_memos.json",
    "sg2": f"{folder_path}/sg2_memos.json",
    "sg3": f"{folder_path}/sg3_memos.json",
    "sg4": f"{folder_path}/sg4_memos.json",
    "sg5": f"{folder_path}/sg5_memos.json",
}

def find_common_words(documents, min_occurrences=2):
    """Find words that appear in all documents of a group."""
    if not documents:
        return []
    
    # Collect all tokens from all documents
    all_tokens = [token for doc in documents for token in doc.get("tokens", [])]
    token_counts = Counter(all_tokens)
    
    # Find words that appear at least min_occurrences times
    common_words = [
        word for word, count in token_counts.items()
        if count >= min_occurrences
    ]
    
    return sorted(common_words)

def process_json_file(json_path, label):
    """Process a single JSON file and return common words."""
    if not os.path.isfile(json_path):
        print(f"[WARNING] File not found: {json_path}")
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if data is a list or a dict with 'documents' key
        documents = data.get("documents", data) if isinstance(data, dict) else data
        if not documents:
            print(f"[WARNING] No documents found in {json_path}")
            return None
        
        common_words = find_common_words(documents)
        return common_words
    except Exception as e:
        print(f"[ERROR] Error processing {json_path}: {e}")
        return None

def process_json_files(json_files_map):
    """Process all JSON files sequentially."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for label, filepath in json_files_map.items():
        common_words = process_json_file(filepath, label)
        if common_words is not None:
            results.append((label, common_words))
    return results

if __name__ == "__main__":
    all_results = process_json_files(json_files)
    
    for label, common_words in all_results:
        output_file = f"{output_dir}/{label}_memos_with_common.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(common_words, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved {len(common_words)} common words for {label} to {output_file}")
