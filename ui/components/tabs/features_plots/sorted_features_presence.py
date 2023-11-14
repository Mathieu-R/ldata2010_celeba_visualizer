import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc

from ui import ids
from dash_extensions.enrich import DashProxy, Output, Input, dcc, callback

@callback(
	Output(ids.HISTOGRAM_SORTED_FEATURES_PRESENCE, "figure"),

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

def render(app: DashProxy) -> dcc.Graph:
	return dcc.Graph(ids.HISTOGRAM_SORTED_FEATURES_PRESENCE)