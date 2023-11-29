import dash_bootstrap_components as dbc 

from dash_extensions.enrich import DashProxy
from .dr import DR
from .clustering import Clustering

def render(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardBody([
			dbc.Row([
				dbc.Col(DR(app).render(), width=6),
				dbc.Col(Clustering(app).render(), width=6)
			])
		])
	])