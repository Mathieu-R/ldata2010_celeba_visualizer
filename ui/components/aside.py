import pandas as pd
import dash_mantine_components as dmc

from dash_extensions.enrich import DashProxy, Serverside, Input, Output, callback, html
from dash_iconify import DashIconify

from ui import ids
from ui.components import features_selection
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
	return html.Div(
		className="aside",
		children=[
			html.H2(children="CelebA dataset visualizer"),
			html.Hr(),
			dmc.Select(
				label="Pick the dataset you want",
				placeholder="Dataset",
				id=ids.DATASET_DROPDOWN,
				value=DATASETS[0],
				data=DATASETS,
				icon=DashIconify(icon="material-symbols-light:dataset-outline"), 
				className="header__select_dataset"
			),
			features_selection.render(app)
		]
	)