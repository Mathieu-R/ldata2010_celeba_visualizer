import dash_mantine_components as dmc

from dash_extensions.enrich import DashProxy, Input, Output, callback, dbc, html
from dash_iconify import DashIconify

from .tabs.photo_gallery import gallery
from .tabs.correlation_matrixes import correlation_matrix_features
from .tabs.features_plots import features_histogram, sorted_features_presence
from ui import ids

def render(app: DashProxy) -> html.Div:
	return html.Div([
		dmc.Tabs([
			dmc.TabsList([
				dmc.Tab(
					"Gallery",
					icon=DashIconify(icon="tabler:photo"),
					value="photo_gallery__tab"
				),
				dmc.Tab(
					"Correlation matrix",
					icon=DashIconify(icon="carbon:scatter-matrix"),
					value="correlation_matrix__tab"
				),
				dmc.Tab(
					"Features plots",
					icon=DashIconify(icon="ep:histogram"),
					value="features_plot__tab"
				),
				dmc.Tab(
					"DR / Clustering",
					icon=DashIconify(icon="octicon:graph-24"),
					value="dr_clustering__tabl"
				)
			]),
			dmc.TabsPanel([
					gallery.render(app)
				],
				value="photo_gallery__tab",
				style={
					"display": "flex",
					"flex": 1,
					"padding": "10px"
				},
				className="photo_gallery__tab"
			),
			dmc.TabsPanel(
				children=correlation_matrix_features.render(app),
				value="correlation_matrix__tab",
				className="correlation_matrix__tab"
			),
			dmc.TabsPanel([
					sorted_features_presence.render(app),
					features_histogram.render(app)
				],
				value="correlation_matrix__tab",
				className="correlation_matrix__tab"
			),
			dmc.TabsPanel(
				children=[],
				value="dr_clustering__tab",
				className="dr_clustering__tab"
			)],
			style={
				"display": "flex",
				"flex-direction": "column",
				"flex": 1
			},
			value="photo_gallery__tab",
			className="tab__container"
		)],
		style={
			"display": "flex",
			"flex": 1
		},
		className=".layout__container"
	)