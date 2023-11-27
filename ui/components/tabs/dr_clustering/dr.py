import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, ALL, dcc, html 
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE

from ui import ids
from ui.components.inputs import input_number_field, input_select_field
from ui.components.graph import custom_graph

from enum import Enum

class DR_ALGO(Enum):
	PCA = "pca"
	TSNE = "tsne"

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
	Output(ids.DR__GRAPH, "figure"),

	Input(ids.DR__START_ALGO, "n_clicks"),
	Input(ids.DR__SELECT_ALGO, "value"),

	Input({"type": ids.DR_TSNE__CONFIG, "index": ALL}, "value"),
	Input(ids.DR__COMPONENTS, "value"),

	Input(ids.FEATURES_STORE, "data"),
	suppress_callback_exceptions=True
)
def start_algo(
	n_clicks: int | None, 
	algo: str, 
	tsne_config: list[int],
	n_components: int,
	features: pd.DataFrame
):
	if n_clicks is None:
		return no_update
	
	n_components = int(n_components)
	
	if algo == DR_ALGO.TSNE.value and len(tsne_config) > 0:
		tsne = TSNE(n_components=n_components, perplexity=float(tsne_config[0]), learning_rate=float(tsne_config[1]), n_iter=tsne_config[2])
		res = tsne.fit_transform(features)
	else:
		pca = PCA(n_components=n_components)
		res = pca.fit_transform(features)
	
	if (n_components == 2):
		return px.scatter(res, x=res[:,0], y=res[:,1])
	else:
		return px.scatter_3d(res, x=res[:,0], y=res[:,1], z=res[:,2])


def dimension_reduction(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader("Dimension Reduction"),
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
