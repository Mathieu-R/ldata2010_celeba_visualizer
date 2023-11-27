import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, ALL, dcc, html 
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans

from ui import ids
from ui.components.inputs import input_number_field, input_select_field
from ui.components.graph import custom_graph

from enum import Enum

class CLUSTERING_ALGO(Enum):
	KMEANS = "k_means"

@callback(
	Output(ids.CLUSTERING__CONFIG, "children"),

	Input(ids.CLUSTERING__SELECT_ALGO, "value")
)
def set_config(algo: str):
	if algo == CLUSTERING_ALGO.KMEANS.value:
		return [
			dbc.Col(input_number_field("Number of clusters", id={"type": ids.CLUSTERING_KMEANS__CONFIG, "index": "clustering_kmeans__n_clusters"}, min=5.0, max=50.0, value=30.0), width=3)
		]
	
@callback(
	Output(ids.CLUSTERING__GRAPH, "figure"),

	Input(ids.CLUSTERING__START_ALGO, "n_clicks"),
	Input(ids.CLUSTERING__SELECT_ALGO, "value"),

	Input({"type": ids.CLUSTERING_KMEANS__CONFIG, "index": ALL}, "value"),

	Input(ids.FEATURES_STORE, "data"),
	suppress_callback_exceptions=True
)
def start_algo(
	n_clicks: int | None, 
	algo: str, 
	kmeans_config: list[int],
	features: pd.DataFrame
):
	if n_clicks is None:
		return no_update
	
	# reduce dimension
	pca = PCA(n_components=2)
	res = pca.fit_transform(features)

	features["pca_1"] = res[:,0]
	features["pca_2"] = res[:,1]

	n_clusters = kmeans_config[0]

	if algo == CLUSTERING_ALGO.KMEANS.value:
		kmeans = KMeans(n_init="auto", n_clusters=n_clusters)
		features["cluster"] = kmeans.fit_predict(features)

	return px.scatter(features, x=features["pca_1"], y=features["pca_2"], labels=features["cluster"])

def clustering(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader("Clustering"),
		dbc.CardBody([
			custom_graph(id=ids.CLUSTERING__GRAPH, no_margin=True)
		]),
		dbc.CardFooter([
			dbc.Row([
				input_select_field(
					title="Algorithm", 
					id=ids.CLUSTERING__SELECT_ALGO,
					options=[
						{"label": CLUSTERING_ALGO.KMEANS.name, "value": CLUSTERING_ALGO.KMEANS.value}
					],
					value=CLUSTERING_ALGO.KMEANS.value
				)
			]),
			html.Br(),
			dbc.Row(id=ids.CLUSTERING__CONFIG),
			html.Br(),
			dbc.Row([dbc.Button("Start", id=ids.CLUSTERING__START_ALGO)])
		])
	])
