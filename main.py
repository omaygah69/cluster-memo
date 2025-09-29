import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from gensim.models import Doc2Vec  # For Doc2Vec embeddings
from gensim.utils import simple_preprocess  # For tokenization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA  # Replaced UMAP with PCA
import os
from scipy.optimize import linear_sum_assignment
import shutil
from gensim.models.doc2vec import TaggedDocument

num_clusters = 5
top_n_representatives = 3
top_n_keywords = 20  # Increased for better matching

memo_stopwords = [
    "student", "students", "school", "memo", "memorandum",
    "office", "university", "college", "campus", "department",
    "faculty", "member", "personnel", "program", "psu", "lingayen",
    "meeting", "activity", "subject", "advisory", "date", "class",
    "committee", "event", "agenda", "office", "official"
]

with open("memos.json", "r", encoding="utf-8") as f:
    memos = json.load(f)

documents = [memo["clean_text"] for memo in memos]
filenames = [memo["filename"] for memo in memos]

# Doc2Vec embeddings
print("Generating document embeddings with Doc2Vec...")
# Prepare tagged documents (required for Doc2Vec)
tagged_docs = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(documents)]
# Train Doc2Vec model
model = Doc2Vec(
    vector_size=384,  # Match MiniLM dimensionality
    window=5,         # Context window size
    min_count=2,      # Ignore rare words
    workers=4,        # Parallel training
    epochs=20,        # Training iterations
    dm=0,             # DBOW mode (faster, good for larger texts)
    seed=42
)
model.build_vocab(tagged_docs)
model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
# Infer vectors for all documents
X = np.array([model.infer_vector(simple_preprocess(doc)) for doc in documents])
X = Normalizer().fit_transform(X)

# PCA reduction (replaced UMAP)
print("Reducing dimensionality with PCA...")
pca_reducer = PCA(
    n_components=60,  # Retain more information, same as before
    random_state=42
)
X_reduced = pca_reducer.fit_transform(X)

# Kmeans cluster
print(f"\nClustering into {num_clusters} clusters using KMeans...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X_reduced)

# Save cluster labels
for memo, label in zip(memos, labels):
    memo["cluster"] = int(label)

# Helper
def get_representative_docs(X, labels, filenames, top_n=3):
    reps = {}
    for cluster_id in set(labels):
        cluster_points = np.where(labels == cluster_id)[0]
        cluster_center = X[cluster_points].mean(axis=0)
        distances = np.linalg.norm(X[cluster_points] - cluster_center, axis=1)
        closest_idxs = cluster_points[np.argsort(distances)[:top_n]]
        reps[cluster_id] = [filenames[idx] for idx in closest_idxs]
    return reps

def summarize_cluster(cluster_id, docs, top_n=5):
    if not docs:
        return f"Cluster {cluster_id}: No clear theme."
    
    vectorizer_local = TfidfVectorizer(
        stop_words=memo_stopwords, max_features=5000, ngram_range=(1,3)
    )
    X_local = vectorizer_local.fit_transform(docs).toarray()
    terms = vectorizer_local.get_feature_names_out()
    cluster_mean = X_local.mean(axis=0)
    top_indices = cluster_mean.argsort()[::-1][:top_n]
    keywords = [terms[i] for i in top_indices]

    sentence1 = f"This cluster is mainly about {', '.join(keywords[:3])}."
    sentence2 = f"Other notable topics include {', '.join(keywords[3:])}."
    return f"Cluster {cluster_id} ({len(docs)} documents): {sentence1} {sentence2}"

# debug
print(f"\nFound {num_clusters} clusters\n")

# Representative docs
representatives = get_representative_docs(X_reduced, labels, filenames, top_n=top_n_representatives)
for cluster, docs in representatives.items():
    print(f"\nCluster {cluster} representative memos:")
    for doc in docs:
        print(" -", doc)

# Cluster summaries
for cluster_id in range(num_clusters):
    cluster_docs = [documents[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
    print("\n" + summarize_cluster(cluster_id, cluster_docs))

# Silhouette scores
if len(set(labels)) > 1:
    global_score = silhouette_score(X_reduced, labels)
    print(f"\nGlobal Silhouette Score: {global_score:.4f}")
    sample_scores = silhouette_samples(X_reduced, labels)
    for cluster_id in set(labels):
        cluster_scores = sample_scores[labels == cluster_id]
        print(f"Cluster {cluster_id}: silhouette = {cluster_scores.mean():.4f}")
else:
    print("\nNot enough clusters for silhouette scores")

# tsne
print("\nRunning t-SNE dimensionality reduction for visualization...")
tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
reduced = tsne.fit_transform(X_reduced)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    reduced[:, 0], reduced[:, 1], reduced[:, 2],
    c=labels, cmap="tab10", alpha=0.7
)
ax.set_title(f"3D t-SNE Visualization (KMeans, {num_clusters} clusters)")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
plt.colorbar(scatter, ax=ax, label="Cluster ID")
plt.show()

# Match clusters to SGs using common words
print("\nMatching clusters to SGs...")
sgs = ["sg1", "sg2", "sg3", "sg4", "sg5"]
keys_dir = "./data_sg/keys"
sg_common = {}
for sg in sgs:
    key_path = f"{keys_dir}/{sg}_memos_with_common.json"
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            sg_common[sg] = set(json.load(f))
    else:
        print(f"[WARNING] Common words file not found for {sg}")
        sg_common[sg] = set()

cluster_keywords = {}
for cluster_id in range(num_clusters):
    cluster_docs = [documents[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
    if not cluster_docs:
        cluster_keywords[cluster_id] = set()
        continue
    vectorizer = TfidfVectorizer(
        stop_words=memo_stopwords, max_features=5000, ngram_range=(1,3)
    )
    X_local = vectorizer.fit_transform(cluster_docs).toarray()
    terms = vectorizer.get_feature_names_out()
    cluster_mean = X_local.mean(axis=0)
    top_indices = cluster_mean.argsort()[::-1][:50]  # Use top 50 for better matching
    keywords = [terms[i] for i in top_indices]
    cluster_keywords[cluster_id] = set(keywords)

# Compute similarity matrix (Jaccard)
sim_matrix = np.zeros((num_clusters, len(sgs)))
for i in range(num_clusters):
    for j, sg in enumerate(sgs):
        inter = len(cluster_keywords.get(i, set()) & sg_common.get(sg, set()))
        union = len(cluster_keywords.get(i, set()) | sg_common.get(sg, set()))
        sim_matrix[i, j] = inter / union if union > 0 else 0

# Assign clusters to SGs
cost = -sim_matrix
row_ind, col_ind = linear_sum_assignment(cost)
cluster_to_sg = {row: sgs[col] for row, col in zip(row_ind, col_ind)}

print("Cluster to SG mapping:")
for cluster, sg in cluster_to_sg.items():
    print(f"Cluster {cluster} -> {sg}")

# Organize memos into results folder
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
for cluster_id, sg in cluster_to_sg.items():
    sg_dir = f"{results_dir}/{sg}"
    os.makedirs(sg_dir, exist_ok=True)
    cluster_memos = [memos[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
    copied_count = 0
    for memo in cluster_memos:
        original_path = os.path.join("./memos", memo["relpath"])
        if os.path.exists(original_path):
            dest_path = os.path.join(sg_dir, memo["filename"])
            shutil.copy(original_path, dest_path)
            copied_count += 1
        else:
            print(f"[WARNING] Original file not found: {original_path}")
    print(f"Copied {copied_count} memos for {sg} to {sg_dir}")
