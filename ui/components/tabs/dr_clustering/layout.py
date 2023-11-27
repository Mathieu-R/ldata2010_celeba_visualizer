import dash_bootstrap_components as dbc 

from dash_extensions.enrich import DashProxy
from .dr import dimension_reduction
from .clustering import clustering

def render(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardBody([
			dbc.Row([
				dbc.Col(dimension_reduction(app), width=6),
				dbc.Col(clustering(app), width=6)
			])
		])
	])