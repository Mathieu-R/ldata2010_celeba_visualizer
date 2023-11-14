import pandas as pd

def precompute(filename: str) -> None:
	df = pd.read_csv(f"data/{filename}.csv")

	df_embeddings = df_l.select_dtypes("float")
	df_features = df_l.select_dtypes("int").drop(columns=["id"])
	df_images = df_l.loc[:, ["image_name", "id"]]

	df_embeddings.to_parquet(f"data/{filename}__embeddings.gzip")
	df_features.to_parquet(f"data/{filename}__features.gzip")
	df_images.to_parquet(f"data/{filename}__images.gzip")


precompute(filename="celeba_buffalo_l")
precompute(filename="celeba_buffalo_s")