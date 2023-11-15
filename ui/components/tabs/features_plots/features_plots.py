import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc

from ui import ids
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
	)

	return fig


@callback(
	Output(ids.FEATURES_PLOTS__HISTS, "children"),

	Input(ids.FEATURES_STORE, "data"),
	Input(ids.FEATURES_SELECTION__SELECT, "value")
)
def feature_plot(features: pd.DataFrame, selected_features: list[str] | None) -> list[dcc.Graph]:
	if selected_features is None:
		raise PreventUpdate
	
	return [
		dcc.Graph(
			figure= px.histogram(
				features[selected_feature].replace({-1: "no", 1: "yes"}),
				title=f"Histogram of {selected_feature}"
			)
		)
		for selected_feature in selected_features
	]

def render(app: DashProxy) -> html.Div:
	return html.Div(
		className="features_plots__container",
		children=[
			dcc.Graph(id=ids.FEATURES_PLOTS__SORTED_FEATURES_PRESENCE_BAR),
			html.Div(
				className="features_plots_hists__grid",
				id=ids.FEATURES_PLOTS__HISTS
			)
		]
	)