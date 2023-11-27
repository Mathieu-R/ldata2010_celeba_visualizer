import dash_bootstrap_components as dbc  

from dash_extensions.enrich import dcc, html

def render(app: DashProxy, title: str, id: str) -> html.Div:
	return html.Div([
		dbc.Button(
			"Close",
			size="sm",
			outline=True
		),
		dbc.Modal([
			dbc.ModalHeader(
				dbc.ModalTitle(title),
				close_button=True
			),
			dbc.ModalBody(
				dcc.Graph(
					id=f"{id}-modal-graph",
					style={"max-height": "none", "height": "80%"}
				)
			)
		])
	])