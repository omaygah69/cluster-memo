import os
import json
from docx import Document
import spacy

nlp = spacy.load("en_core_web_sm")

folder_path = "./memos"
subfolders = {
    "ccs": f"{folder_path}/Ccs",
    "bsed": f"{folder_path}/Bsed",
    "ssc": f"{folder_path}/ssc",
    "casl": f"{folder_path}/Casl",
    "cbpa": f"{folder_path}/Cbpa",
    "hrm": f"{folder_path}/HM",
    "sg": f"{folder_path}/sg",
    "permissions": f"{folder_path}/permissions",
    "strategies": f"{folder_path}/Strategies",
}

domain_stopwords = {
    "student", "students", "school", "memo", "memorandum",
    "office", "university", "college", "campus", "department",
    "faculty", "member", "personnel", "program", "psu", "lingayen"
}

def extract_clean_text(docx_path):
    docx = Document(docx_path)
    text = "\n".join([para.text for para in docx.paragraphs])
    doc = nlp(text)

    clean_tokens = [
        token.lemma_.lower().strip()
        for token in doc
        if token.lemma_ not in ["-PRON-"]
        and not token.is_stop
        and not token.is_punct
        and not token.like_num
        and token.is_alpha
        and token.lemma_.lower() not in domain_stopwords
        and token.ent_type_ not in {"DATE", "TIME", "PERSON", "ORG", "GPE", "FAC", "NORP"}
        and token.pos_ in {"NOUN", "ADJ"}
    ]

    clean_text = " ".join(clean_tokens)
    sentences = [sent.text.strip() for sent in doc.sents]

    return {
        "filename": os.path.basename(docx_path),
        "relpath": os.path.relpath(docx_path, folder_path),
        "raw_text": text,
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
                if file.lower().endswith(".docx"):
                    path = os.path.join(root, file)
                    data = extract_clean_text(path)
                    data["folder"] = label
                    results.append(data)
    return results

if __name__ == "__main__":
    memos = process_subfolders(subfolders)
    with open("memos.json", "w", encoding="utf-8") as f:
        json.dump(memos, f, indent=2, ensure_ascii=False)
    print(f"Processed {len(memos)} documents.")
