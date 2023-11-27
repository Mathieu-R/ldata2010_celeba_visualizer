import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html

def input_select_field(title: str, id: str, options, value: str) -> html.Div:
	return html.Div([
		html.P(title),
		html.Div([
			dbc.Select(
				id=id,
				persistence=True,
				persistence_type="local",
				value=value,
				options=options,
				class_name="input_select"
			)
		])
	], className="input_select__container")

def input_dropdown_field(title: str, placeholder: str, id: str) -> html.Div:
	return html.Div([
		html.P(title),
		dcc.Dropdown(
			placeholder=placeholder,
			multi=True,
			id=id,
			style={
				"width": "100%"
			}
		)
	])