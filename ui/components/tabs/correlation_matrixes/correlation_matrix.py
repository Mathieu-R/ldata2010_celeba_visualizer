import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, dcc, html
from dash.exceptions import PreventUpdate
from constants import NUMBER_OF_FEATURES

from ui import ids
from ui.components.graph import custom_graph
from ui.components.inputs import input_dropdown_field, input_select_field
from ui.components.buttons import popout_button

def get_triangle_corr(square_corr: pd.DataFrame) -> pd.DataFrame:
	mask = np.triu(np.ones_like(square_corr, dtype=bool))
	corr = square_corr.mask(mask)
	return corr

@callback(
	Output(ids.CORRELATION_FEATURES__FILTER, "options"),
	Output(ids.CORRELATION_FEATURES_VS_EMBEDDINGS__FILTER, "options"),

	Input(ids.FEATURES_STORE, "data")
)
def set_features_in_select(features: pd.DataFrame | None):	
	if features is None:
		raise PreventUpdate
	
	features_list = features.columns.to_list()

	return (features_list, features_list)

@callback(
	Output(ids.CORRELATION_FEATURES__GRAPH, "figure"),

	Input(ids.FEATURES_STORE, "data"),
	Input(ids.CORRELATION_FEATURES__FILTER, "value")
)
def compute_features_correlation_matrix(features: pd.DataFrame | None, selected_features: list[str] | None):
	if features is None:
		raise PreventUpdate
	
	if selected_features is None:
		corr = features.corr().abs()
	elif len(selected_features) == 1:
		raise PreventUpdate
	else:
		corr = features[selected_features].corr().abs()

	fig = px.imshow(get_triangle_corr(corr))
	return fig

# @callback(
# 	Output(ids.CORRELATION_EMBEDDINGS__SELECT_PARTS, "options"),

# 	Input(ids.EMBEDDINGS_STORE, "data")
# )
# def set_embeddings_parts_in_select(embeddings: pd.DataFrame | None) -> list[str]:
# 	n_columns = embeddings.columns.shape[0]
# 	return [f"{i}-{min(i+10, n_columns)}" for i in np.arange(0, n_columns, 10)]

@callback(
	Output(ids.CORRELATION_EMBEDDINGS__GRAPH, "figure"),

	Input(ids.EMBEDDINGS_STORE, "data"),
	Input(ids.CORRELATION_EMBEDDINGS__SELECT_PARTS, "value")
)
def compute_embeddings_correlation_matrix(embeddings: pd.DataFrame, parts: list[int]):
	#if parts_str == "":
	#	raise PreventUpdate
	
	#parts = parts_str.split("-")
	start = int(parts[0])
	stop = int(parts[1])

	corr = embeddings.iloc[:,start:stop].corr().abs()
	return px.imshow(get_triangle_corr(corr))

@callback(
 	Output(ids.CORRELATION_FEATURES_VS_EMBEDDINGS__GRAPH, "figure"),

	Input(ids.FEATURES_STORE, "data"),
 	Input(ids.EMBEDDINGS_STORE, "data"),
	 
	Input(ids.CORRELATION_FEATURES_VS_EMBEDDINGS__FILTER, "value"),
	Input(ids.CORRELATION_FEATURES_VS_EMBEDDINGS__SELECT_PARTS, "value")
)
def compute_features_vs_embeddings_correlation_matrix(features: pd.DataFrame, embeddings: pd.DataFrame, selected_features: list[str] | None, parts: list[int]):
	start = parts[0]
	stop = parts[1]

	if selected_features is not None:
		features = features[selected_features]

	embeddings = embeddings.iloc[:,start:stop]

	# fast way
	features = features - features.mean()
	embeddings = embeddings - embeddings.mean()

	corr = features.T.dot(embeddings).div(len(features)).div(embeddings.std(ddof=0)).div(features.std(ddof=0), axis=0)

	return px.imshow(corr)

def correlation_matrix_features_card(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader([
			html.H5("Features correlation matrix")
		]),
		dbc.CardBody([
			custom_graph(
				id=ids.CORRELATION_FEATURES__GRAPH,
			)
		]),
		dbc.CardFooter([
			input_dropdown_field(
				title="Filter features", 
				placeholder="You need to select at least 2 features",
				id=ids.CORRELATION_FEATURES__FILTER
			)
		])
	])

def correlation_matrix_embeddings_card(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader([
			html.H5("Embeddings correlation matrix"),
			# input_select_field(
			# 	title="Embeddings",
			# 	id=ids.CORRELATION_EMBEDDINGS__SELECT_PARTS,
			# 	options=[],
			# 	value=""
			# )
		]),
		dbc.CardBody(
			custom_graph(id=ids.CORRELATION_EMBEDDINGS__GRAPH),
		),
		dbc.CardFooter([
			dbc.Row([
				dbc.Col(
					dcc.RangeSlider(
						min=0, 
						max=512, 
						value=[0, 10],
						allowCross=False, 
						tooltip={"placement": "bottom", "always_visible": False},
						id=ids.CORRELATION_EMBEDDINGS__SELECT_PARTS
					)
				)
			])
		])
	])

def correlation_matrix_features_vs_embeddings_card(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader([
			html.H5("Features vs Embeddings correlation matrix")
		]),
		dbc.CardBody([
			custom_graph(
				id=ids.CORRELATION_FEATURES_VS_EMBEDDINGS__GRAPH,
			)
		]),
		dbc.CardFooter([
			dbc.Row([
				dbc.Col(
					input_dropdown_field(
						title="Filter features", 
						placeholder="You need to select at least 2 features",
						id=ids.CORRELATION_FEATURES_VS_EMBEDDINGS__FILTER
					)
				)
			]),
			dbc.Row([
				dbc.Col(
					dcc.RangeSlider(
						min=0, 
						max=512, 
						value=[0, 10],
						allowCross=False, 
						tooltip={"placement": "bottom", "always_visible": False},
						id=ids.CORRELATION_FEATURES_VS_EMBEDDINGS__SELECT_PARTS
					)
				)
			])
		])
	])

def render(app: DashProxy) -> html.Div:
	return html.Div([
		dbc.Card([
			dbc.CardBody([
				dbc.Row([
					dbc.Col(correlation_matrix_features_card(app), width=6),
					dbc.Col(correlation_matrix_embeddings_card(app), width=6)
				]),
				dbc.Row([
					dbc.Col(correlation_matrix_features_vs_embeddings_card(app))
				])
			])
		], color="dark")
	])