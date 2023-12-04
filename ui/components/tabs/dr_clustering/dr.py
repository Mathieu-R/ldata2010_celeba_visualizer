import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import hashlib

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, ALL, dcc, html
from dash.exceptions import PreventUpdate
from plotly import graph_objs as go
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from openTSNE import TSNE
from umap import UMAP 

from typing import Any

from ui import ids
from ui.components.inputs import input_number_field, input_select_field
from ui.components.graph import custom_graph

from enum import Enum

class DR_ALGO(Enum):
	PCA = "pca"
	TSNE = "tsne"

class DATA(Enum):
	FEATURES = "features"
	EMBEDDINGS = "embeddings"

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
					dbc.Col(input_number_field("Learning rate", id={"type": ids.DR_TSNE__CONFIG, "index": ids.DR_TSNE__LEARNING_RATE}, min=10.0, max=1000.0, value=200.0), width=3),
					dbc.Col(input_number_field("Number of iterations", id={"type": ids.DR_TSNE__CONFIG, "index": ids.DR_TSNE__ITERATIONS}, min=250, max=50000, value=1000), width=4)
			]
			
		@callback(
			output=Output(ids.DR__GRAPH, "figure"),

			inputs=[
				Input(ids.DR__START_ALGO, "n_clicks"),
				Input(ids.DR__SELECT_DATA, "value"),
				Input(ids.DR__SELECT_ALGO, "value"),

				Input({"type": ids.DR_TSNE__CONFIG, "index": ALL}, "value"),
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
			dataset_sample, dataset_rest = dataset.loc[indices[:3000],:], dataset.loc[indices[3000:],:]
			

			if algo == DR_ALGO.TSNE.value and len(tsne_config) > 0:
				config = dict(n_components=n_components, perplexity=tsne_config[0], learning_rate=tsne_config[1], n_iter=tsne_config[2])
				fig = self.compute_and_save(algo=algo, algo_fun=TSNE, config=config, type="full", dataset=dataset)
				return fig

			else:
				config = dict(n_components=n_components)
				fig = self.compute_and_save(algo=algo, algo_fun=PCA, config=config, type="subset", dataset=dataset_sample)
				set_progress(fig)

				fig = self.compute_and_save(algo=algo, algo_fun=PCA, config=config, type="full", dataset=dataset)
				return fig
		
	def compute_and_save(self, algo: str, algo_fun: Any, config: dict[str, Any], type: str, dataset: pd.DataFrame):
		hash = self.hash_config(
			algo=algo,
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

		if (config["n_components"] == 2):
			return px.scatter(res, x=res[:,0], y=res[:,1])
		else:
			return px.scatter_3d(res, x=res[:,0], y=res[:,1], z=res[:,2])
		
	def hash_config(self, algo: str, type: str, config: dict[str, Any]) -> str:
		input = algo + type + str(sorted(config.items()))
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
							{"label": "TSNE", "value": DR_ALGO.TSNE.value}
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
