import pandas as pd
import dash_mantine_components as dmc

from dash_extensions.enrich import Dash, Input, Output, callback
from ui import ids

@callback(
	Output(ids.FEATURES_SELECTION__SELECT, "data"),

	Input(ids.FEATURES_STORE, "data")
)
def show_gallery_on_data(features: pd.DataFrame) -> list[str]:	
	return features.columns.to_list()

def render(app: Dash) -> dmc.MultiSelect:
	return dmc.MultiSelect(
		placeholder="Select all the features you desire",
		id=ids.FEATURES_SELECTION__SELECT,
		className="features_selection__select"
	)