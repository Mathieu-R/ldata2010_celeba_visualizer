import pandas as pd
import dash_mantine_components as dmc

from dash_extensions.enrich import DashProxy, Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from PIL import Image

from ui import ids

PICTURES_BY_PAGE = 50

@callback(
	Output(ids.PHOTO_GALLERY__SLIDER, "max"),

	Input(ids.IMAGES_STORE, "data"),
	Input(ids.FEATURES_STORE, "data"),
	Input(ids.FEATURES_SELECTION__SELECT, "value")
)
def compute_slider_properties(images: pd.DataFrame, features: pd.DataFrame, selected_features: list[str] | None) -> int:	
	if selected_features is None:
		raise PreventUpdate
	
	features_mask = features[selected_features].isin([1]).all(axis=1)

	subset_images = images[features_mask]
	
	n_images = subset_images.shape[0]
	max = int(n_images / PICTURES_BY_PAGE)

	return max

@callback(
	Output(ids.PHOTO_GALLERY__GRID, "children"),

	Input(ids.IMAGES_STORE, "data"),
	Input(ids.FEATURES_STORE, "data"),
	Input(ids.FEATURES_SELECTION__SELECT, "value"),
	Input(ids.PHOTO_GALLERY__SLIDER, "value")
)
def show_gallery(images_dict: dict, features_dict: dict, selected_features: list[str] | None, page: int) -> list[dmc.Image]:	
	if selected_features is None:
		raise PreventUpdate
	
	images = pd.DataFrame(images_dict)
	features = pd.DataFrame(features_dict)
	features_mask = features[selected_features].isin([1]).all(axis=1)

	subset_images = images[features_mask]
	
	offset_start = (page - 1) * PICTURES_BY_PAGE
	offset_stop = offset_start + PICTURES_BY_PAGE

	return [
		dmc.Image(
			src=Image.open(f"assets/img_celeba/{image_name}"),
			alt=image_name,
			radius="sm",
			width=100, height=100,
			withPlaceholder=True,
			className="photo_gallery__image"
		) for image_name in subset_images.loc[:, "image_name"][offset_start:offset_stop]
	]

def render(app: DashProxy) -> html.Div:
	return html.Div([
		html.Div(
			style={
				"display": "flex",
				"flex-wrap": "wrap",
				"justify-content": "flex_start"
			},
			id=ids.PHOTO_GALLERY__GRID
		),
		dcc.Slider(
			min=1,
			step=1, 
			max=1, 
			id=ids.PHOTO_GALLERY__SLIDER, 
			className="photo_gallery__slider"
		)], 
		style={
			"display": "flex",
			"flex": 1,
			"flex-direction": "column"
		},
		className="photo_gallery__container"
	)