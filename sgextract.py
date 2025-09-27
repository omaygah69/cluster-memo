import os
import json
import fitz  # PyMuPDF
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5_000_000  # allow up to 5M chars

folder_path = "./memos"
subfolders = {
    "sg1": f"{folder_path}/collecting-sg/Sg1",
    "sg2": f"{folder_path}/collecting-sg/Sg2",
    "sg3": f"{folder_path}/collecting-sg/Sg3",
    "sg4": f"{folder_path}/collecting-sg/Sg4",
    "sg5": f"{folder_path}/collecting-sg/Sg5",
}

domain_stopwords = {
    "student", "students", "school", "memo", "memorandum",
    "office", "university", "college", "campus", "department",
    "faculty", "member", "personnel", "program", "psu", "lingayen"
}

def extract_clean_text(pdf_path):
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]

    # process each page separately
    clean_tokens = []
    sentences = []
    raw_text = "\n".join(texts)

    for page_text in texts:
        spacy_doc = nlp(page_text)

        tokens = [
            token.lemma_.lower().strip()
            for token in spacy_doc
            if token.lemma_ not in ["-PRON-"]
            and not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.is_alpha
            and token.lemma_.lower() not in domain_stopwords
            and token.ent_type_ not in {"DATE", "TIME", "PERSON", "ORG", "GPE", "FAC", "NORP"}
            and token.pos_ in {"NOUN", "ADJ"}
        ]

        clean_tokens.extend(tokens)
        sentences.extend([sent.text.strip() for sent in spacy_doc.sents])

    clean_text = " ".join(clean_tokens)

    return {
        "filename": os.path.basename(pdf_path),
        "relpath": os.path.relpath(pdf_path, folder_path),
        "raw_text": raw_text,
        "clean_text": clean_text,
        "tokens": clean_tokens,
        "sentences": sentences,
    }


def process_subfolders(subfolders_map):
    results = []
    for label, dirpath in subfolders_map.items():
        if not os.path.isdir(dirpath):
            continue
        for root, _, files in os.walk(dirpath):
            for file in files:
                if file.lower().endswith(".pdf"):
                    path = os.path.join(root, file)
                    data = extract_clean_text(path)
                    data["folder"] = label
                    results.append(data)
    return results

if __name__ == "__main__":
    memos = process_subfolders(subfolders)

    # group by folder name (sg1, sg2, …)
    grouped = {}
    for memo in memos:
        folder = memo["folder"]
        grouped.setdefault(folder, []).append(memo)

    # make sure output directory exists
    out_dir = "./data_sg"
    os.makedirs(out_dir, exist_ok=True)

    # save each SG into its own JSON file
    for folder, docs in grouped.items():
        out_file = os.path.join(out_dir, f"{folder}_memos.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(docs)} documents to {out_file}")

    # optional: also save a combined JSON with everything
    combined_file = os.path.join(out_dir, "sg_all_memos.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(memos, f, indent=2, ensure_ascii=False)
    print(f"Saved combined {len(memos)} documents to {combined_file}")
