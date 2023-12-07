import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import numpy.typing as npt
import pandas as pd
import hashlib

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, ALL, dcc, html 
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from hdbscan import HDBSCAN
from typing import Any

from ui import ids
from ui.components.inputs import input_number_field, input_select_field, input_radio_field
from ui.components.graph import custom_graph

from enum import Enum

class PROJECTION(Enum):
	TSNE = "tsne_exaggeration_12"
	UMAP = "umap"

class CLUSTERING_ALGO(Enum):
	MINIBATCHKMEANS = "mini_batch_k_means"
	HDBSCAN = "hdb_scan"
	AGGLOMERATIVECLUSTERING = "agglometrative_clustering"

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
				Input(ids.CLUSTERING__SELECT_DATA, "value"),
				Input(ids.CLUSTERING__PROJECTION, "value"),
				Input(ids.CLUSTERING__SELECT_ALGO, "value"),

				Input({"type": ids.CLUSTERING_KMEANS__CONFIG, "index": ALL}, "value"),
				Input({"type": ids.CLUSTERING_DBSCAN__CONFIG, "index": ALL}, "value"),
				Input({"type": ids.CLUSTERING_AGGLOMERATIVE__CONFIG, "index": ALL}, "value"),

				Input(ids.DATASET_DROPDOWN, "value"),
				Input(ids.FEATURES_STORE, "data"),
				Input(ids.EMBEDDINGS_STORE, "data"),
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
			features: pd.DataFrame,
			embeddings: pd.DataFrame
		):
			if n_clicks is None:
				raise PreventUpdate
			
			if data_category == DATA.FEATURES.value:
				dataset = features
			else:
				dataset = embeddings

			data_projected = np.load(f"precomputed/{dataset_file}__{data_category}_{projection}.npy")

			print(algo)

			if algo == CLUSTERING_ALGO.MINIBATCHKMEANS.value:
				n_clusters = kmeans_config[0]
				config = dict(n_init="auto", n_clusters=n_clusters)

				fig = self.compute_and_save(algo=algo, algo_fun=MiniBatchKMeans, config=config, dataset_file=dataset_file, data_category=data_category, type=DATA_PART.FULL.value, projection=projection, dataset=dataset, data_projected=data_projected)
				return fig

			elif algo == CLUSTERING_ALGO.HDBSCAN.value:
				min_cluster_size = int(dbscan_config[0])
				min_samples = int(dbscan_config[1])

				config = dict(min_cluster_size=min_cluster_size, min_samples=min_samples)

				fig = self.compute_and_save(algo=algo, algo_fun=HDBSCAN, config=config, dataset_file=dataset_file, data_category=data_category, type=DATA_PART.FULL.value, projection=projection, dataset=dataset, data_projected=data_projected)
				return fig
			else:
				n_clusters = kmeans_config[0]
				linkage = agglomerative_config[0]
				config = dict(n_clusters=n_clusters, linkage=linkage)

				fig = self.compute_and_save(algo=algo, algo_fun=AgglomerativeClustering, dataset_file=dataset_file, data_category=data_category, config=config, type=DATA_PART.FULL.value, projection=projection, dataset=dataset, data_projected=data_projected)
				return fig

	def compute_and_save(self, algo: str, algo_fun: Any, config: dict[str, Any], dataset_file: str, data_category: str, type: str, projection: str, dataset: pd.DataFrame, data_projected: pd.DataFrame):
		hash = self.hash_config(
			algo=algo,
			data_category=data_category,
			type=type,
			projection=projection,
			config=config
		)

		dataset, data_projected = self.get_sample_if_needed(algo, dataset, data_projected)

		# if value has already been computed
		if hash in self._memo:
			labels = self._memo[hash]
		# compute the value and save it for future use
		else:
			instance = self.get_algo_instance(algo, algo_fun, config)
			labels = instance.fit_predict(dataset)
			self._memo[hash] = labels

		data = pd.DataFrame(data_projected, columns=["pc1", "pc2", "pc3"])
		data["cluster"] = labels

		# to have cluster as discrete values in legend
		data["cluster"] = data["cluster"].astype(str)

		return px.scatter(data, x="pc1", y="pc2", color="cluster", labels={"cluster": "Cluster"}, opacity=0.5, color_discrete_sequence=px.colors.qualitative.Plotly)
	
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

	def hash_config(self, algo: str, data_category: str, type: str, projection: str, config: dict[str, Any]) -> str:
		input = algo + data_category + type + projection + str(sorted(config.items()))
		hash = hashlib.sha256(input.encode())
		hash_hex = hash.hexdigest()
		return hash_hex

	def render(self) -> dbc.Card:
		return dbc.Card([
			dbc.CardHeader([
				"Clustering",
				dbc.Row([
					dbc.Col(
						input_select_field(
							title=None,
							id=ids.CLUSTERING__SELECT_DATA,
							options=[
								{"label": DATA.FEATURES.name, "value": DATA.FEATURES.value},
								{"label": DATA.EMBEDDINGS.name, "value": DATA.EMBEDDINGS.value}
							],
							value=DATA.FEATURES.value
						),
						width=9
					),
					dbc.Col(
						input_radio_field(
							title="Projection",
							id=ids.CLUSTERING__PROJECTION,
							options=[
								{"label": PROJECTION.TSNE.name, "value": PROJECTION.TSNE.value}, {"label": PROJECTION.UMAP.name, "value": PROJECTION.UMAP.value}
							],
							value=PROJECTION.UMAP.value
						),
						width=3
					)
				])
				
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
