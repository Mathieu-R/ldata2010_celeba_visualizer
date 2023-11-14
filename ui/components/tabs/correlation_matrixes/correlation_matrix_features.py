import plotly.express as px
import pandas as pd
import numpy as np

from dash_extensions.enrich import DashProxy, Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from ui import ids

def get_triangle_corr(square_corr: pd.DataFrame) -> pd.DataFrame:
	mask = np.triu(np.ones_like(square_corr, dtype=bool))
	corr = square_corr.mask(mask)
	return corr

@callback(
	Output(ids.CORRELATION_FEATURES__GRAPH, "figure"),

	Input(ids.FEATURES_STORE, "data"),
	Input(ids.FEATURES_SELECTION__SELECT, "value")
)
def compute_features_correlation_matrix(features: pd.DataFrame, selected_features: list[str] | None):
	if features is None:
		raise PreventUpdate
	
	if selected_features is None:
		corr = features.corr().abs()
	else:
		corr = features[selected_features].corr().abs()

	fig = px.imshow(get_triangle_corr(corr))
	return fig

@callback(
	Output(ids.CORRELATION_EMBEDDINGS__GRAPH, "figure"),

	Input(ids.EMBEDDINGS_STORE, "data")
)
def compute_embeddings_correlation_matrix(embeddings: pd.DataFrame):
	corr = embeddings.corr().abs()
	return px.imshow(get_triangle_corr(corr))

@callback(
	Output(ids.CORRELATION_FEATURES_VS_EMBEDDINGS__GRAPH, "figure"),

	Input(ids.FEATURES_STORE, "data"),
	Input(ids.EMBEDDINGS_STORE, "data"),
	Input(ids.FEATURES_SELECTION__SELECT, "value")
)
def compute_features_vs_embeddings_correlation_matrix(features: pd.DataFrame, embeddings: pd.DataFrame,  selected_features: list[str] | None):
	if selected_features is None:
		corr = features.corrwith(embeddings).abs()
	else:
		corr = features[selected_features].corrwith(embeddings).abs()
	
	return px.imshow(get_triangle_corr(corr))

def render(app: DashProxy) -> list[dcc.Graph]:
	return [
		dcc.Graph(id=ids.CORRELATION_FEATURES__GRAPH),
		dcc.Graph(id=ids.CORRELATION_EMBEDDINGS__GRAPH),
		dcc.Graph(id=ids.CORRELATION_FEATURES_VS_EMBEDDINGS__GRAPH),
	]