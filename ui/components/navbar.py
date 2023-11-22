import dash_mantine_components as dmc

from dash_extensions.enrich import DashProxy, Input, Output, callback, dcc, html
from dash_iconify import DashIconify

from .tabs.photo_gallery import gallery
from .tabs.correlation_matrixes import correlation_matrix_features
from .tabs.features_plots import features_plots
from ui import ids

def render(app: DashProxy) -> html.Div:
	return html.Div(
		className=".layout__container",
		style={
			"display": "flex",
			"flex": 1
		},
		children=[
		dcc.Tabs(
			parent_className="tab__content",
			className="tab__container",
			value="photo_gallery__tab",
			children=[
			dcc.Tab([
					gallery.render(app)
				],
				label="Gallery",
				value="photo_gallery__tab",
				style={
					"display": "flex",
					"flex": 1,
					"padding": "10px"
				},
				className="photo_gallery__tab"
			),
			dcc.Tab(
				children=correlation_matrix_features.render(app),
				label="Correlation Matrix",
				value="correlation_matrix__tab",
				className="correlation_matrix__tab"
			),
			dcc.Tab([
					features_plots.render(app)
				],
				label="Features Plots",
				value="features_plot__tab",
				className="features_plot__tab"
			),
			dcc.Tab(
				children=[],
				label="DR / Clustering",
				value="dr_clustering__tab",
				className="dr_clustering__tab"
			)]
		)]
	)