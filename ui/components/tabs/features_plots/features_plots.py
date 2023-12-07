import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

from ui import ids
from ui.components import inputs, graph
from dash_extensions.enrich import DashProxy, Output, Input, dcc, callback, html
from dash.exceptions import PreventUpdate

@callback(
	Output(ids.FEATURES_PLOTS__SORTED_FEATURES_PRESENCE_BAR, "figure"),

	Input(ids.FEATURES_STORE, "data")
)
def feature_presence_counter_plot(features: pd.DataFrame):
	counts = pd.DataFrame({col: features[col].value_counts() for col in features})
	counts = counts.iloc[1]
	counts = counts.sort_values(ascending=False)
	counts = pd.DataFrame(counts).reset_index()
	counts.columns = ["Feature", "Count"]

	fig = px.bar(
		counts, 
		title="Most present features",
		x="Feature",
		y="Count"
	).update_layout(
		template='plotly_dark',
		plot_bgcolor= 'rgba(0, 0, 0, 0)',
		paper_bgcolor= 'rgba(0, 0, 0, 0)',
	)

	return fig

def most_present_features(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader(
			html.H5("Most Present Features")
		),
		dbc.CardBody(
			graph.custom_graph(id=ids.FEATURES_PLOTS__SORTED_FEATURES_PRESENCE_BAR)
		)
	])

@callback(
	Output(ids.FEATURES_PLOTS__SELECT_FEATURE, "options"),

	Input(ids.FEATURES_STORE, "data")
)
def set_features_in_select(features: pd.DataFrame) -> list[str]:
	features_list = features.columns.to_list()
	return features_list

@callback(
	Output(ids.FEATURES_PLOTS__HIST, "figure"),

	Input(ids.FEATURES_STORE, "data"),
	Input(ids.FEATURES_PLOTS__SELECT_FEATURE, "value")
)
def feature_plot(features: pd.DataFrame, selected_feature: str | None):
	if selected_feature is None or selected_feature == "":
		raise PreventUpdate
	
	fig = px.histogram(
		features[selected_feature].map({-1: "No", 1: "Yes"}),
		title=f"Histogram of {selected_feature}"
	).update_layout(
		template='plotly_dark',
		plot_bgcolor= 'rgba(0, 0, 0, 0)',
		paper_bgcolor= 'rgba(0, 0, 0, 0)',
	)

	return fig


def histograms(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader([
			html.H5("Histogram of each feature"),
			inputs.input_select_field(
				title="Feature",
				id=ids.FEATURES_PLOTS__SELECT_FEATURE,
				options=[],
				value=""
			)
		]),
		dbc.CardBody(
			graph.custom_graph(id=ids.FEATURES_PLOTS__HIST)
		),
		dbc.CardFooter([
			
		])
	])

def render(app: DashProxy) -> html.Div:
	return html.Div([
		dbc.Card([
			dbc.CardBody([
				dbc.Row([
					dbc.Col([most_present_features(app)], width=6),
					dbc.Col([histograms(app)], width=6)
				], align="center")
			])
		], color="dark")
	])