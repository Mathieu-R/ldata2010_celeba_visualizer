import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import numpy.typing as npt
import pandas as pd
import hashlib

from dash_extensions.enrich import DashProxy, Input, Output, State, callback, ALL, dcc, html 
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from hdbscan import HDBSCAN
from typing import Any

from ui import ids
from ui.components.inputs import input_number_field, input_select_field, input_radio_field
from ui.components.graph import custom_graph

from enum import Enum

class PROJECTION(Enum):
	TSNE = "tsne"
	UMAP = "umap"

class CLUSTERING_ALGO(Enum):
	MINIBATCHKMEANS = "mini_batch_k_means"
	HDBSCAN = "hdb_scan"
	AGGLOMERATIVECLUSTERING = "agglomerative_clustering"

class AGGLOMERATIVE_LINKAGE(Enum):
	WARD = "ward"
	COMPLETE = "complete"
	AVERAGE = "average"
	SINGLE = "single"

class DATA(Enum):
	FEATURES = "features"
	EMBEDDINGS = "embeddings"

class DATA_PART(Enum):
	SAMPLE = "sample"
	REST = "rest"
	FULL = "full"

ALGOS_CONFIG = {
	CLUSTERING_ALGO.MINIBATCHKMEANS.value: {
		# run algo on full dataset
		"samples": None
	},
	CLUSTERING_ALGO.HDBSCAN.value: {
		# too slow to run on full dataset
		# (~ 10min)
		"samples": 5000
	},
	CLUSTERING_ALGO.AGGLOMERATIVECLUSTERING.value: {
		# too slow to run on full dataset
		"samples": 10000
	}
}

class Clustering:
	def __init__(self, app: DashProxy) -> None:
		self._app = app
		self._memo = {}

		if hasattr(self, "callbacks"):
			self.callbacks(app)

	def callbacks(self, app: DashProxy):
		@callback(
			Output(ids.CLUSTERING__SELECT_INFORMATIVE_SAMPLE, "options"),

			Input(ids.FEATURES_STORE, "data")
		)
		def set_features_in_select(features: pd.DataFrame) -> list[str]:
			features_list = features.columns.to_list()
			return features_list

		@callback(
			Output(ids.CLUSTERING__CONFIG, "children"),

			Input(ids.CLUSTERING__SELECT_ALGO, "value")
		)
		def set_config(algo: str):
			if algo == CLUSTERING_ALGO.MINIBATCHKMEANS.value:
				return [
					dbc.Col(input_number_field("Number of clusters", id={"type": ids.CLUSTERING_KMEANS__CONFIG, "index": "clustering_kmeans__n_clusters"}, min=2, max=30, value=3), width=3)
				]
			elif algo == CLUSTERING_ALGO.HDBSCAN.value:
				return [
					dbc.Col(input_number_field("Min cluster size", id={"type": ids.CLUSTERING_DBSCAN__CONFIG, "index": "clustering_dbscan__min_cluster_size"}, min=1, max=100, value=5), width=3),
					dbc.Col(input_number_field("Min samples", id={"type": ids.CLUSTERING_DBSCAN__CONFIG, "index": "clustering_dbscan__min_samples"}, min=1, max=1000, value=5), width=3)
				]
			else:
				return [
					dbc.Col(input_number_field("Number of clusters", id={"type": ids.CLUSTERING_KMEANS__CONFIG, "index": "clustering_kmeans__n_clusters"}, min=2, max=30, value=3), width=3),
					dbc.Col(
						input_select_field(
							"Linkage", 
							id={"type": ids.CLUSTERING_AGGLOMERATIVE__CONFIG, "index": "clustering_agglomerative__linkage"},
							options=[
								{"label": AGGLOMERATIVE_LINKAGE.WARD.name, "value": AGGLOMERATIVE_LINKAGE.WARD.value},
								{"label": AGGLOMERATIVE_LINKAGE.COMPLETE.name, "value": AGGLOMERATIVE_LINKAGE.COMPLETE.value},
								{"label": AGGLOMERATIVE_LINKAGE.AVERAGE.name, "value": AGGLOMERATIVE_LINKAGE.AVERAGE.value},
								{"label": AGGLOMERATIVE_LINKAGE.SINGLE.name, "value": AGGLOMERATIVE_LINKAGE.SINGLE.value}
							],
							value=AGGLOMERATIVE_LINKAGE.WARD.value
						)
					)
				]
			
		@callback(
			output=Output(ids.CLUSTERING__GRAPH, "figure"),
			inputs=[
				Input(ids.CLUSTERING__START_ALGO, "n_clicks"),
				State(ids.CLUSTERING__SELECT_DATA, "value"),
				State(ids.CLUSTERING__PROJECTION, "value"),
				State(ids.CLUSTERING__SELECT_ALGO, "value"),

				State({"type": ids.CLUSTERING_KMEANS__CONFIG, "index": ALL}, "value"),
				State({"type": ids.CLUSTERING_DBSCAN__CONFIG, "index": ALL}, "value"),
				State({"type": ids.CLUSTERING_AGGLOMERATIVE__CONFIG, "index": ALL}, "value"),

				State(ids.DATASET_DROPDOWN, "value"),

				State(ids.CLUSTERING__SELECT_INFORMATIVE_SAMPLE, "value"),
				State(ids.FEATURES_STORE, "data"),
				State(ids.EMBEDDINGS_STORE, "data"),
			],
			running=[
				(Output(ids.CLUSTERING__START_ALGO, "disabled"), True, False)
			],
			progess_default=go.Figure(),
			progress=Output(ids.CLUSTERING__GRAPH, "figure"),
			background=True,
			suppress_callback_exceptions=True
		)
		def start_algo(
			set_progress: Any,
			n_clicks: int | None, 
			data_category: str,
			projection: str,
			algo: str, 
			kmeans_config: list[int],
			dbscan_config: list[float],
			agglomerative_config: list[int],
			dataset_file: str,
			selected_feature: str,
			features: pd.DataFrame,
			embeddings: pd.DataFrame
		):
			if n_clicks is None:
				raise PreventUpdate
			
			if data_category == DATA.FEATURES.value:
				dataset = features
			else:
				# reduce to 50 dimensions for embeddings through PCA
				# it is ok to use this reduced dimension dataset for clustering
				dataset = np.load(f"precomputed/{dataset_file}__{data_category}_pca_50.npy")

			data_projected = np.load(f"precomputed/{dataset_file}__{data_category}_{projection}.npy")

			print(algo)

			if algo == CLUSTERING_ALGO.MINIBATCHKMEANS.value:
				n_clusters = kmeans_config[0]
				config = dict(n_init="auto", n_clusters=n_clusters)

				fig = self.compute_and_save(algo=algo, algo_fun=MiniBatchKMeans, config=config, dataset_file=dataset_file, data_category=data_category, type=DATA_PART.FULL.value, projection=projection, features=features, selected_feature=selected_feature, dataset=dataset, data_projected=data_projected)
				return fig

			elif algo == CLUSTERING_ALGO.HDBSCAN.value:
				min_cluster_size = int(dbscan_config[0])
				min_samples = int(dbscan_config[1])

				config = dict(min_cluster_size=min_cluster_size, min_samples=min_samples)

				fig = self.compute_and_save(algo=algo, algo_fun=HDBSCAN, config=config, dataset_file=dataset_file, data_category=data_category, type=DATA_PART.FULL.value, projection=projection, features=features, selected_feature=selected_feature, dataset=dataset, data_projected=data_projected)
				return fig
			else:
				n_clusters = kmeans_config[0]
				linkage = agglomerative_config[0]
				config = dict(n_clusters=n_clusters, linkage=linkage)

				fig = self.compute_and_save(algo=algo, algo_fun=AgglomerativeClustering, dataset_file=dataset_file, data_category=data_category, config=config, type=DATA_PART.FULL.value, projection=projection, features=features, selected_feature=selected_feature, dataset=dataset, data_projected=data_projected)
				return fig

	def compute_and_save(self, algo: str, algo_fun: Any, config: dict[str, Any], dataset_file: str, data_category: str, type: str, projection: str, features: pd.DataFrame, selected_feature: str, dataset: pd.DataFrame, data_projected: pd.DataFrame):
		hash = self.hash_config(
			algo=algo,
			data_category=data_category,
			type=type,
			selected_feature=selected_feature,
			projection=projection,
			config=config
		)

		dataset_informative_sample, data_projected_informative_sample = self.get_informative_sample(
			features=features, 
			selected_feature=selected_feature,
			dataset=dataset, 
			data_projected=data_projected
		)

		# if value has already been computed
		if hash in self._memo:
			labels = self._memo[hash]
		# compute the value and save it for future use
		else:
			instance = self.get_algo_instance(algo, algo_fun, config)
			labels = instance.fit_predict(dataset_informative_sample)
			self._memo[hash] = labels

		columns = self.get_columns(projection)

		data = pd.DataFrame(data_projected_informative_sample, columns=columns)
		data["cluster"] = labels

		# to have cluster as discrete values in legend
		data["cluster"] = data["cluster"].astype(str)

		return px.scatter(data, x=columns[0], y=columns[1], color="cluster", opacity=0.5, color_discrete_sequence=px.colors.qualitative.Plotly)
	
	def get_informative_sample(self, features: pd.DataFrame, selected_feature: str, dataset: pd.DataFrame, data_projected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
		features_mask = features[[selected_feature]].isin([1]).all(axis=1)
		dataset_informative_sample = dataset[features_mask]
		data_projected_informative_sample = data_projected[features_mask]

		return dataset_informative_sample, data_projected_informative_sample
	
	def get_sample_if_needed(self, algo: str, dataset: pd.DataFrame, data_projected: pd.DataFrame) -> Any:
		n_samples = ALGOS_CONFIG[algo]["samples"]

		if n_samples is None:
			return (dataset, data_projected)
		
		indices = np.random.permutation(list(range(dataset.shape[0])))
		dataset_sample = dataset.loc[indices[:n_samples],:]
		data_projected_sample = np.take(data_projected, indices[:n_samples], axis=0)
		return (dataset_sample, data_projected_sample)
	
	def get_algo_instance(self, algo: str, algo_fun: Any, config: dict[str, Any]) -> Any:
		if algo == CLUSTERING_ALGO.MINIBATCHKMEANS.value:
			instance = algo_fun(**config, random_state=42)
		else:
			instance = algo_fun(**config)
		
		return instance
	
	def get_columns(self, projection: str) -> list[str]:
		if projection == PROJECTION.TSNE.value:
			columns = ["tsne1", "tsne2", "tsne3"]
		else:
			columns = ["umap1", "umap2", "umap3"]
		
		return columns

	def hash_config(self, algo: str, data_category: str, type: str, selected_feature: str, projection: str, config: dict[str, Any]) -> str:
		input = algo + data_category + type + selected_feature + projection + str(sorted(config.items()))
		hash = hashlib.sha256(input.encode())
		hash_hex = hash.hexdigest()
		return hash_hex

	def render(self) -> dbc.Card:
		return dbc.Card([
			dbc.CardHeader([
				"Clustering",
				dbc.Col([
					dbc.Row(
						input_select_field(
							title=None,
							id=ids.CLUSTERING__SELECT_DATA,
							options=[
								{"label": DATA.FEATURES.name, "value": DATA.FEATURES.value},
								{"label": DATA.EMBEDDINGS.name, "value": DATA.EMBEDDINGS.value}
							],
							value=DATA.FEATURES.value
						)
					),
					dbc.Row([
						input_select_field(
							title="Informative sample",
							id=ids.CLUSTERING__SELECT_INFORMATIVE_SAMPLE
						)
					])
				], width=9),
				dbc.Col(
					input_radio_field(
						title="Projection",
						id=ids.CLUSTERING__PROJECTION,
						options=[
							{"label": PROJECTION.TSNE.name, "value": PROJECTION.TSNE.value}, 
							{"label": PROJECTION.UMAP.name, "value": PROJECTION.UMAP.value}
						],
						value=PROJECTION.UMAP.value
					),
					width=3
				)
			]),
			dbc.CardBody([
				custom_graph(id=ids.CLUSTERING__GRAPH, no_margin=True)
			]),
			dbc.CardFooter([
				dbc.Row([
					input_select_field(
						title="Algorithm", 
						id=ids.CLUSTERING__SELECT_ALGO,
						options=[
							{"label": "Mini Batch KMeans", "value": CLUSTERING_ALGO.MINIBATCHKMEANS.value},
							{"label": "HDBScan", "value": CLUSTERING_ALGO.HDBSCAN.value},
							{"label": "Agglomerative Clustering", "value": CLUSTERING_ALGO.AGGLOMERATIVECLUSTERING.value},
						],
						value=CLUSTERING_ALGO.MINIBATCHKMEANS.value
					)
				]),
				html.Br(),
				dbc.Row(id=ids.CLUSTERING__CONFIG),
				html.Br(),
				dbc.Row([dbc.Button("Start", id=ids.CLUSTERING__START_ALGO)])
			])
		])
