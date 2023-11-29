import pandas as pd
import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate

from ui import ids
from ui.components import inputs

PICTURES_BY_PAGE = 50

def get_subset_images(images: pd.DataFrame, features: pd.DataFrame, selected_features: list[str] | None) -> pd.DataFrame:
	if selected_features is None:
		subset_images = images
	else:
		features_mask = features[selected_features].isin([1]).all(axis=1)
		subset_images = images[features_mask]

	return subset_images

@callback(
	Output(ids.PHOTO_GALLERY__FILTER, "options"),

	Input(ids.FEATURES_STORE, "data")
)
def set_features_in_select(features: pd.DataFrame | None) -> list[str]:	
	return features.columns.to_list()

@callback(
	Output(ids.PHOTO_GALLERY__PAGINATION, "max_value"),

	Input(ids.IMAGES_STORE, "data"),
	Input(ids.FEATURES_STORE, "data"),
	Input(ids.PHOTO_GALLERY__FILTER, "value")
)
def compute_pagination_properties(images: pd.DataFrame, features: pd.DataFrame, selected_features: list[str] | None) -> int:	
	subset_images = get_subset_images(images, features, selected_features)
	
	n_images = subset_images.shape[0]
	max = int(n_images / PICTURES_BY_PAGE)

	return max

@callback(
	Output(ids.PHOTO_GALLERY__GRID, "children"),

	Input(ids.IMAGES_STORE, "data"),
	Input(ids.FEATURES_STORE, "data"),
	Input(ids.PHOTO_GALLERY__FILTER, "value"),
	Input(ids.PHOTO_GALLERY__PAGINATION, "active_page")
)
def show_gallery(images: pd.DataFrame, features: pd.DataFrame, selected_features: list[str] | None, page: int) -> list[html.Img]:	
	subset_images = get_subset_images(images, features, selected_features)
	
	offset_start = (page - 1) * PICTURES_BY_PAGE
	offset_stop = offset_start + PICTURES_BY_PAGE

	return [
		html.Img(
			src=f"assets/img_celeba/{image_name}",
			alt=image_name,
			width=100, height=100,
			className="photo_gallery__image"
		) for image_name in subset_images.loc[:, "image_name"][offset_start:offset_stop]
	]

def render(app: DashProxy) -> html.Div:
	return html.Div([
		dbc.Card([
			dbc.CardHeader([
				html.H5("Photo Gallery"),
				inputs.input_dropdown_field(
					title="Filter features", 
					placeholder="Filter the features",
					id=ids.PHOTO_GALLERY__FILTER
				)
			]),
			dbc.CardBody([
				html.Div(
					style={
						"display": "flex",
						"flex-wrap": "wrap",
						"justify-content": "flex_start"
					},
					id=ids.PHOTO_GALLERY__GRID
				)
			]),
			dbc.CardFooter([
				dbc.Pagination(
					min_value=1,
					active_page=1,
					max_value=10,
					fully_expanded=False, 
					id=ids.PHOTO_GALLERY__PAGINATION, 
					className="photo_gallery__pagination"
				)
			])
		], color="dark")
	])