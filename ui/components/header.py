import pandas as pd
import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, Serverside, Input, Output, callback, html

from ui import ids
from ui.components.inputs import input_select_field
from constants import DATASETS

@callback(
	Output(ids.FEATURES_STORE, "data"), 
	Output(ids.EMBEDDINGS_STORE, "data"),
	Output(ids.IMAGES_STORE, "data"),

	Input(ids.DATASET_DROPDOWN, "value")
)
def load_dataset(dataset: str):
	features = pd.read_parquet(f"data/{dataset}__features.gzip")
	images = pd.read_parquet(f"data/{dataset}__images.gzip")
	embeddings = pd.read_parquet(f"data/{dataset}__embeddings.gzip")

	# can only store JSON data in dcc Store
	return Serverside(features), Serverside(embeddings), Serverside(images)

def render(app: DashProxy) -> html.Div:
	return html.Div([
		dbc.Row([
			dbc.Col([html.H2(children="CelebA dataset visualizer")]),
			dbc.Col([
				dbc.Select(
					id=ids.DATASET_DROPDOWN,
					persistence=True,
					persistence_type="local",
					value=DATASETS[0],
					options=DATASETS,
					class_name="input_select"
				)
			])
		])
	], style={"padding": "10px 5px"})