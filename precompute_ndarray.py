import numpy as np
import pandas as pd

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def precompute_ndarray(filename: str) -> None:
	df_features = pd.read_parquet(f"data/{filename}__features.gzip")
	df_embeddings = pd.read_parquet(f"data/{filename}__embeddings.gzip")

	precompute_pca_for_clustering(filename, data_category="embeddings", df=df_embeddings)

	precompute_umap_for_clustering(filename, data_category="features", df=df_features)
	precompute_umap_for_clustering(filename, data_category="embeddings", df=df_embeddings)

	# precompute_tsne_for_clustering(filename, data_category="features", df=df_features)
	# precompute_tsne_for_clustering(filename, data_category="embeddings", df=df_embeddings)

	precompute_tsne_for_clustering_exaggeration_12(filename, data_category="features", df=df_features)
	precompute_tsne_for_clustering_exaggeration_12(filename, data_category="embeddings", df=df_embeddings)

def precompute_pca_for_clustering(filename: str, data_category: str, df: pd.DataFrame) -> None:
	print(f"PRECOMPUTING - filename: {filename}, type: {data_category}, algo: PCA")
	pca = PCA(n_components=50)

	res_pca = pca.fit_transform(df)
	np.save(f"precomputed/{filename}__{data_category}_pca_50.npy", res_pca)
	print("DONE.")

def precompute_umap_for_clustering(filename: str, data_category: str, df: pd.DataFrame) -> None:
	print(f"PRECOMPUTING - filename: {filename}, type: {data_category}, algo: UMAP")
	umap = UMAP(n_components=3, min_dist=1.0)

	# only for embeddings
	# high dimension => 50 dim. with PCA (we can run clustering on it)
	if data_category == "embeddings":
		df = np.load(f"precomputed/{filename}__{data_category}_pca_50.npy")

	# max 50 dim. => 3 dim. with UMAP (for data visualization)
	res_umap = umap.fit_transform(df)
	np.save(f"precomputed/{filename}__{data_category}_umap.npy", res_umap)
	print("DONE.")

# def precompute_tsne_for_clustering(filename: str, data_category: str, df: pd.DataFrame) -> None:
# 	print(f"PRECOMPUTING - filename: {filename}, type: {data_category}, algo: TSNE")
# 	tsne = TSNE(n_components=3, perplexity=50, early_exaggeration=30, random_state=42)
# 	res = tsne.fit_transform(df)
# 	np.save(f"precomputed/{filename}__{data_category}_tsne.npy", res)
# 	print("DONE.")

def precompute_tsne_for_clustering_exaggeration_12(filename: str, data_category: str, df: pd.DataFrame) -> None:
	print(f"PRECOMPUTING - filename: {filename}, type: {data_category}, algo: TSNE, exaggeration: 12")
	tsne = TSNE(n_components=3, perplexity=50, random_state=42)

	# only for embeddings
	# high dimension => 50 dim. with PCA (we can run clustering on it)
	if data_category == "embeddings":
		df = np.load(f"precomputed/{filename}__{data_category}_pca_50.npy")

	# max 50 dim. => 3 dim. with UMAP (for data visualization)
	res_tsne = tsne.fit_transform(df)
	np.save(f"precomputed/{filename}__{data_category}_tsne.npy", res_tsne)
	print("DONE.")

precompute_ndarray(filename="celeba_buffalo_l")
precompute_ndarray(filename="celeba_buffalo_s")