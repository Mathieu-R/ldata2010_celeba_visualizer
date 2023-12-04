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
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn_extra.cluster import CLARA
from typing import Any

from ui import ids
from ui.components.inputs import input_number_field, input_select_field
from ui.components.graph import custom_graph

from enum import Enum

class CLUSTERING_ALGO(Enum):
	MINIBATCHKMEANS = "mini_batch_k_means"
	KMEDOIDS = "k_medoids"
	AGGLOMERATIVECLUSTERING = "agglometrative_clustering"

class AGGLOMERATIVE_LINKAGE(Enum):
	WARD = "ward"
	COMPLETE = "complete"
	AVERAGE = "average"
	SINGLE = "single"

class DATA(Enum):
	FEATURES = "features"
	EMBEDDINGS = "embeddings"

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
			if algo == CLUSTERING_ALGO.MINIBATCHKMEANS.value or algo == CLUSTERING_ALGO.KMEDOIDS.value:
				return [
					dbc.Col(input_number_field("Number of clusters", id={"type": ids.CLUSTERING_KMEANS__CONFIG, "index": "clustering_kmeans__n_clusters"}, min=2, max=30, value=3), width=3)
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
				Input(ids.DR__SELECT_DATA, "value"),
				Input(ids.CLUSTERING__SELECT_ALGO, "value"),

				Input({"type": ids.CLUSTERING_KMEANS__CONFIG, "index": ALL}, "value"),
				Input({"type": ids.CLUSTERING_AGGLOMERATIVE__CONFIG, "index": ALL}, "value"),

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
			algo: str, 
			kmeans_config: list[int],
			agglomerative_config: list[int],
			features: pd.DataFrame,
			embeddings: pd.DataFrame
		):
			if n_clicks is None:
				raise PreventUpdate
			
			if data_category == DATA.FEATURES.value:
				dataset = features
			else:
				dataset = embeddings

			indices = np.random.permutation(list(range(dataset.shape[0])))
			dataset_sample, dataset_rest = dataset.loc[indices[:3000],:], dataset.loc[indices[3000:],:]

			n_clusters = kmeans_config[0]

			if algo == CLUSTERING_ALGO.MINIBATCHKMEANS.value:
				config = dict(n_init="auto", n_clusters=n_clusters)

				fig = self.compute_and_save(algo=algo, algo_fun=MiniBatchKMeans, config=config, data_category=data_category, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=MiniBatchKMeans, config=config, data_category=data_category, type="full", dataset=dataset)
				return fig

			elif algo == CLUSTERING_ALGO.KMEDOIDS.value:
				config = dict(n_clusters=n_clusters)
				fig = self.compute_and_save(algo=algo, algo_fun=KMedoids, config=config, data_category=data_category, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=KMedoids, config=config, data_category=data_category, type="full", dataset=dataset)
				return fig
			else:
				linkage = agglomerative_config[0]
				config = dict(n_clusters=n_clusters, linkage=linkage)
				fig = self.compute_and_save(algo=algo, algo_fun=AgglomerativeClustering, data_category=data_category, config=config, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=AgglomerativeClustering, data_category=data_category, config=config, type="full", dataset=dataset)
				return fig

	def compute_and_save(self, algo: str, algo_fun: Any, config: dict[str, Any], data_category: str, type: str, dataset: pd.DataFrame):
		hash = self.hash_config(
			algo=algo,
			data_category=data_category,
			type=type,
			config=config
		)

		# if value has already been computed
		if hash in self._memo:
			res = self._memo[hash]["res"]
			labels = self._memo[hash]["labels"]
		# compute the value and save it for future use
		else:
			# reduce dimension
			#scaler = StandardScaler()
			#scaled = scaler.fit_transform(dataset)
			tsne = TSNE(n_components=2, perplexity=50, random_state=42)
			res = tsne.fit_transform(dataset)
			#pca = PCA(n_components=2)
			#res = pca.fit_transform(scaled)

			if algo == CLUSTERING_ALGO.AGGLOMERATIVECLUSTERING.value:
				instance = algo_fun(**config)
			else:
				instance = algo_fun(**config, random_state=42)

			if algo == CLUSTERING_ALGO.KMEDOIDS.value:
				labels = instance.fit(dataset).labels_
			else:
				labels = instance.fit_predict(dataset)

			self._memo[hash] = {"res": res, "labels": labels}

		data = pd.DataFrame(res, columns=["pc1", "pc2"])
		data["cluster"] = labels

		# to have cluster as discret values in legend
		data["cluster"] = data["cluster"].astype(str)

		custom_colors = px.colors.qualitative.Plotly#[:config["n_clusters"]]

		return px.scatter(data, x="pc1", y="pc2", color="cluster", labels={"cluster": "Cluster"}, opacity=0.5, color_discrete_sequence=custom_colors)
		
	def hash_config(self, algo: str, data_category: str, type: str, config: dict[str, Any]) -> str:
		input = algo + data_category + type + str(sorted(config.items()))
		hash = hashlib.sha256(input.encode())
		hash_hex = hash.hexdigest()
		return hash_hex

	def render(self) -> dbc.Card:
		return dbc.Card([
			dbc.CardHeader([
				"Clustering",
				input_select_field(
					title=None,
					id=ids.CLUSTERING__SELECT_DATA,
					options=[
						{"label": DATA.FEATURES.name, "value": DATA.FEATURES.value},
						{"label": DATA.EMBEDDINGS.name, "value": DATA.EMBEDDINGS.value}
					],
					value=DATA.FEATURES.value
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
							{"label": "KMedoids", "value": CLUSTERING_ALGO.KMEDOIDS.value},
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
