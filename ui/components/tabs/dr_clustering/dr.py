import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import hashlib

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, ALL, dcc, html
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go

from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from umap import UMAP 

from typing import Any

from ui import ids
from ui.components.inputs import input_number_field, input_select_field
from ui.components.graph import custom_graph

from enum import Enum

class DR_ALGO(Enum):
	PCA = "pca"
	TSNE = "tsne"
	UMAP = "umap"

class DATA(Enum):
	FEATURES = "features"
	EMBEDDINGS = "embeddings"

UMAP_METRICS = [
	"euclidean", "manhattan", "chebyshev", "minkowski",  
	"canberra", "braycurtis", "haversine",
	"mahalanobis", "wminkowski", "seuclidean",
	"cosine", "correlation"
	"hamming", "jaccard", "dice", "russellrao", "kulsinski", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule"
]

class DATA_PART(Enum):
	SAMPLE = "sample"
	REST = "rest"
	FULL = "full"

ALGOS_CONFIG = {
	DR_ALGO.PCA.value: {
		# run algo on full dataset
		"samples": None
	},
	DR_ALGO.TSNE.value: {
		# too slow to run on full dataset
		# (~ 30min)
		"samples": 1000
	},
	DR_ALGO.UMAP.value: {
		# too slow to run on full dataset
		"samples": 1000
	}
}

class DR:
	def __init__(self, app: DashProxy) -> None:
		self._app = app
		self._memo = {}

		if hasattr(self, "callbacks"):
			self.callbacks(self._app)

	def callbacks(self, app: DashProxy):
		@callback(
			Output(ids.DR__CONFIG, "children"),

			Input(ids.DR__SELECT_ALGO, "value")
		)
		def set_config(algo: str):
			if algo == DR_ALGO.TSNE.value:
				return [
					dbc.Col(input_number_field("Perplexity", id={"type": ids.DR_TSNE__CONFIG, "index": ids.DR_TSNE__PERPLEXITY}, min=5.0, max=50.0, value=30.0), width=3),
					dbc.Col(input_number_field("Exaggeration", id={"type": ids.DR_TSNE__CONFIG, "index": ids.DR_TSNE__EXAGGERATION}, min=1.0, max=100.0, value=12.0), width=3),
					dbc.Col(input_number_field("Number of iterations", id={"type": ids.DR_TSNE__CONFIG, "index": ids.DR_TSNE__ITERATIONS}, min=250, max=50000, value=1000), width=4)
				]
			elif algo == DR_ALGO.UMAP.value:
				return [
					dbc.Col(input_number_field("Neighbors", id={"type": ids.DR_UMAP__CONFIG, "index": ids.DR_UMAP__NEIGHBORS}, min=1, max=200, value=15), width=3),
					dbc.Col(input_number_field("Min. dist", id={"type": ids.DR_UMAP__CONFIG, "index": ids.DR_UMAP__MIN_DIST}, min=0.0, max=1.0, value=0.1), width=3),
					dbc.Col(input_select_field(
						"Metric", 
						id={"type": ids.DR_UMAP__CONFIG, "index": ids.DR_UMAP__METRIC}, 
						options=UMAP_METRICS,
						value=UMAP_METRICS[0]
					), width=3),
				]
			
		@callback(
			output=Output(ids.DR__GRAPH, "figure"),

			inputs=[
				Input(ids.DR__START_ALGO, "n_clicks"),
				Input(ids.DR__SELECT_DATA, "value"),
				Input(ids.DR__SELECT_ALGO, "value"),

				Input({"type": ids.DR_TSNE__CONFIG, "index": ALL}, "value"),
				Input({"type": ids.DR_UMAP__CONFIG, "index": ALL}, "value"),
				Input(ids.DR__COMPONENTS, "value"),

				Input(ids.FEATURES_STORE, "data"),
				Input(ids.EMBEDDINGS_STORE, "data"),
			],
			running=[
				(Output(ids.DR__START_ALGO, "disabled"), True, False)
			],
			progess_default=go.Figure(),
			progress=Output(ids.DR__GRAPH, "figure"),
			background=True,
			suppress_callback_exceptions=True
		)
		def start_algo(
			set_progress: Any,
			n_clicks: int | None, 
			data_category: str,
			algo: str, 
			tsne_config: list[int],
			umap_config: Any,
			n_components: int,
			features: pd.DataFrame,
			embeddings: pd.DataFrame
		):
			if n_clicks is None:
				raise PreventUpdate
			
			n_components = int(n_components)

			if data_category == DATA.FEATURES.value:
				dataset = features
			else:
				dataset = embeddings
			
			indices = np.random.permutation(list(range(dataset.shape[0])))
			reverse = np.argsort(indices)

			dataset_sample, dataset_rest = dataset.loc[indices[:3000],:], dataset.loc[indices[3000:],:]

			if algo == DR_ALGO.TSNE.value:
				config = dict(n_components=n_components, perplexity=tsne_config[0], early_exaggeration=tsne_config[1], n_iter=tsne_config[2])

				fig = self.compute_and_save(algo=algo, algo_fun=TSNE, config=config, data_category=data_category, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=TNSE, config=config, data_category=data_category, type="subset", dataset=dataset)
				return fig

			elif algo == DR_ALGO.UMAP.value:
				config = dict(n_components=n_components, n_neighbors=int(umap_config[0]), min_dist=float(umap_config[1]), metric=umap_config[2])
				print(config)
				fig = self.compute_and_save(algo=algo, algo_fun=UMAP, config=config, data_category=data_category, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=UMAP, config=config, data_category=data_category, type="full", dataset=dataset)
				return fig

			else:
				config = dict(n_components=n_components)
				fig = self.compute_and_save(algo=algo, algo_fun=PCA, config=config, data_category=data_category, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=PCA, config=config, data_category=data_category, type="full", dataset=dataset)
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
			res = self._memo[hash]
		# compute the value and save it for future use
		else:
			instance = algo_fun(**config)
			res = instance.fit_transform(dataset)
				
			self._memo[hash] = res

		return self.get_fig(data=res, n_components=config["n_components"])
		
		
	def get_fig(self, data: Any, n_components: int):
		if n_components == 2:
			return px.scatter(data, x=data[:,0], y=data[:,1])
		else:
			return px.scatter_3d(data, x=data[:,0], y=data[:,1], z=data[:,2])
		
	def hash_config(self, algo: str, data_category: str, type: str, config: dict[str, Any]) -> str:
		input = algo + data_category + type + str(sorted(config.items()))
		hash = hashlib.sha256(input.encode())
		hash_hex = hash.hexdigest()
		return hash_hex

	def render(self) -> dbc.Card:
		return dbc.Card([
			dbc.CardHeader([
				"Dimension Reduction",
				input_select_field(
					title=None,
					id=ids.DR__SELECT_DATA,
					options=[
						{"label": DATA.FEATURES.name, "value": DATA.FEATURES.value},
						{"label": DATA.EMBEDDINGS.name, "value": DATA.EMBEDDINGS.value}
					],
					value=DATA.FEATURES.value
				)
			]),
			dbc.CardBody([
				custom_graph(id=ids.DR__GRAPH, no_margin=True)
			]),
			dbc.CardFooter([
				dbc.Row([
					input_select_field(
						title="Algorithm", 
						id=ids.DR__SELECT_ALGO,
						options=[
							{"label": "PCA", "value": DR_ALGO.PCA.value},
							{"label": "TSNE", "value": DR_ALGO.TSNE.value},
							{"label": "UMAP", "value": DR_ALGO.UMAP.value}
						],
						value=DR_ALGO.PCA.value
					)
				]),
				html.Br(),
				dbc.Row([
					dbc.Row(id=ids.DR__CONFIG),
					dbc.Row([
						dbc.Col(
							input_select_field(
								title="Number of components", 
								id=ids.DR__COMPONENTS, 
								options=[
									{"label": "2", "value": 2},
									{"label": "3", "value": 3}
								],
								value=2
							)
						)
					])
				]),
				html.Br(),
				dbc.Row([dbc.Button("Start", id=ids.DR__START_ALGO)])
			])
		])
