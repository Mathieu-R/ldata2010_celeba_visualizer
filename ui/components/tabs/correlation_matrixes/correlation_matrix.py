import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, dcc, html
from dash.exceptions import PreventUpdate
from constants import NUMBER_OF_FEATURES

from ui import ids
from ui.components.graph import custom_graph
from ui.components.inputs import input_dropdown_field
from ui.components.buttons import popout_button

def get_triangle_corr(square_corr: pd.DataFrame) -> pd.DataFrame:
	mask = np.triu(np.ones_like(square_corr, dtype=bool))
	corr = square_corr.mask(mask)
	return corr

@callback(
	Output(ids.CORRELATION_FEATURES__FILTER, "options"),

	Input(ids.FEATURES_STORE, "data")
)
def set_features_in_select(features: pd.DataFrame | None) -> list[str]:	
	if features is None:
		raise PreventUpdate
	
	return features.columns.to_list()

@callback(
	Output(ids.CORRELATION_FEATURES__GRAPH, "figure"),

	Input(ids.FEATURES_STORE, "data"),
	Input(ids.CORRELATION_FEATURES__FILTER, "value")
)
def compute_features_correlation_matrix(features: pd.DataFrame | None, selected_features: list[str] | None):
	if features is None:
		raise PreventUpdate
	
	if selected_features is None or len(selected_features) == 0:
		corr = features.corr().abs()
	elif len(selected_features) == 1:
		raise PreventUpdate
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

# @callback(
# 	Output(ids.CORRELATION_FEATURES_VS_EMBEDDINGS__GRAPH, "figure"),

# 	Input(ids.FEATURES_STORE, "data"),
# 	Input(ids.EMBEDDINGS_STORE, "data"),
# 	Input(ids.FEATURES_SELECTION__SELECT, "value")
# )
# def compute_features_vs_embeddings_correlation_matrix(features: pd.DataFrame, embeddings: pd.DataFrame,  selected_features: list[str] | None):
# 	if selected_features is None:
# 		corr = features.corrwith(embeddings).abs()
# 	else:
# 		corr = features[selected_features].corrwith(embeddings).abs()
	
	return px.imshow(get_triangle_corr(corr))

def correlation_matrix_features_card(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader([
			html.H5("Features correlation matrix")
		]),
		dbc.CardBody([
			custom_graph(
				id=ids.CORRELATION_FEATURES__GRAPH,
			),
			dbc.Row([
				dbc.Col([
					popout_button(id=ids.CORRELATION_FEATURES__POPOUT)
				], md=2, align="start")
			], justify="end")
		]),
		dbc.CardFooter([
			input_dropdown_field(
				title="Filter features", 
				placeholder="Filter the features",
				id=ids.CORRELATION_FEATURES__FILTER
			)
		])
	])

def correlation_matrix_embeddings_card(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader([
			html.H5("Embeddings correlation matrix")
		]),
		dbc.CardBody(
			custom_graph(id=ids.CORRELATION_EMBEDDINGS__GRAPH),
		),
		dbc.CardFooter([
			
		])
	])

def render(app: DashProxy) -> html.Div:
	return html.Div([
		dbc.Card([
			dbc.CardBody([
				dbc.Row([
					dbc.Col(correlation_matrix_features_card(app), width=6),
					dbc.Col(correlation_matrix_embeddings_card(app), width=6)
				])
			])
		], color="dark")
	])