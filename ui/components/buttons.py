import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, dcc, html 

def popout_button(id: str) -> dbc.Button:
	return dbc.Button("popout", id=id, color="secondary", size="sm", outline=True, class_name="me-1")