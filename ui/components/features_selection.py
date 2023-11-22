import pandas as pd
import dash_mantine_components as dmc

from dash_extensions.enrich import DashProxy, Input, Output, callback, ctx, html
from dash.exceptions import PreventUpdate
from ui import ids

@callback(
	Output(ids.FEATURES_SELECTION__SELECT, "data"),

	Input(ids.FEATURES_STORE, "data")
)
def set_features_in_select(features: pd.DataFrame) -> list[str]:	
	return features.columns.to_list()

@callback(
	Output(ids.FEATURES_SELECTION__SELECT, "value"),
	
	Input(ids.FEATURES_STORE, "data"),
	Input(ids.FEATURES_SELECTION__ALL_BUTTON, "n_clicks"),
	Input(ids.FEATURES_SELECTION__RESET_BUTTON, "n_clicks")
)
def set_features(features: pd.DataFrame, all: int | None, reset: int | None) -> list[str]:	
	# retrieve context
	triggered_id = ctx.triggered_id

	if triggered_id == ids.FEATURES_SELECTION__ALL_BUTTON:
		return features.columns.to_list()
	elif triggered_id == ids.FEATURES_SELECTION__RESET_BUTTON:
		return []

def render(app: DashProxy) -> html.Div:
	return html.Div(
		className="features_selection__container",
		children=[
			dmc.MultiSelect(
				label="Desired features",
				placeholder="Select all the features you desire",
				id=ids.FEATURES_SELECTION__SELECT,
				className="features_selection__select"
			),
			html.Div(
				className="features_selection__buttons",
				children=[
					dmc.Button(
						children="all",
						id=ids.FEATURES_SELECTION__ALL_BUTTON,
						variant="light"
					),
					dmc.Button(
						children="reset",
						id=ids.FEATURES_SELECTION__RESET_BUTTON,
						variant="light"
					)
				]
			)
		]
	)