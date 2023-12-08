from typing import Any

import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html

def input_number_field(title: str | None, id: str | dict[str, str], **args) -> html.Div:
	if (title is not None):
		return html.Div([
			html.P(title),
			html.Div([
				dbc.Input(id=id, type="number", **args)
			])
		])
	else:
		return html.Div([
			html.Div([
				dbc.Input(id=id, type="number", min=min, max=max, value=value)
			])
		])

def input_select_field(title: str | None, id: str | dict[str, str], options: list[dict[str, Any]] | list[str], value: Any) -> html.Div:
	if title is not None:
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
	else:
		return html.Div([
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

def input_dropdown_field(title: str, placeholder: str, id: str | dict[str, str]) -> html.Div:
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

def input_radio_field(title: str, id: str | dict[str, str], options: list[dict[str, Any]] | list[str], value: Any) -> html.Div:
	return html.Div([
		dbc.Label(title),
		dbc.RadioItems(
			id=id,
			options=options,
			value=value
		)
	])