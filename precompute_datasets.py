import numpy as np
import pandas as pd

def precompute(filename: str) -> None:
	df = pd.read_csv(f"data/{filename}.csv")

	df_embeddings = df.select_dtypes("float")
	df_features = df.select_dtypes("int").drop(columns=["id"])
	df_images = df.loc[:, ["image_name", "id"]]

	df_features.to_parquet(f"data/{filename}__features.gzip")
	df_embeddings.to_parquet(f"data/{filename}__embeddings.gzip")
	df_images.to_parquet(f"data/{filename}__images.gzip")


precompute(filename="celeba_buffalo_l")
precompute(filename="celeba_buffalo_s")