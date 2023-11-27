import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, Input, Output, callback, dcc, html
from dash_iconify import DashIconify

from .tabs.photo_gallery import gallery
from .tabs.correlation_matrixes import correlation_matrix
from .tabs.features_plots import features_plots
from .tabs.dr_clustering import layout as dr_clustering_layout
from ui import ids

def render(app: DashProxy) -> html.Div:
	return html.Div(
		className=".layout__container",
		style={
			"display": "flex",
			"flex-direction": "column",
			"flex": 1
		},
		children=[
		dbc.Tabs(
			class_name="tab__container",
			active_tab="photo_gallery__tab",
			children=[
			dbc.Tab([
					gallery.render(app)
				],
				label="Gallery",
				tab_id="photo_gallery__tab",
				class_name="photo_gallery__tab"
			),
			dbc.Tab([
					correlation_matrix.render(app)
				],
				label="Correlation Matrix",
				tab_id="correlation_matrix__tab",
				class_name="correlation_matrix__tab"
			),
			dbc.Tab([
					features_plots.render(app)
				],
				label="Features Plots",
				tab_id="features_plot__tab",
				class_name="features_plot__tab"
			),
			dbc.Tab([
					dr_clustering_layout.render(app)
				],
				label="DR / Clustering",
				tab_id="dr_clustering__tab",
				class_name="dr_clustering__tab"
			)]
		)]
	)