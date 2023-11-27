import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, Input, Output, callback, no_update, dcc, html 
from plotly import graph_objs
from ui import ids

from enum import Enum

class DR_ALGO(Enum):
	PCA = "pca"
	TSNE = "tsne"

dr_layout = graph_objs.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)


def input_number_field(title: str, id: str, min: int | float, max: int | float, value: int | float) -> html.Div:
	return html.Div([
		html.P(title),
		html.Div([
			dbc.Input(id=id, type="number", min=min, max=max, value=value)
		])
	])

def input_select_field(title: str, id: str, options) -> html.Div:
	return html.Div([
		html.P(title),
		html.Div([
			dbc.Select(
				id=id,
				options=options
			)
		])
	], className="input_select__container")

@callback(
	Output(ids.DR__CONFIG, "children"),

	Input(ids.DR__SELECT_ALGO, "value")
)
def set_config(algo: str):
	if algo == DR_ALGO.TSNE:
		return [
			input_number_field("Perplexity", id=ids.DR_TSNE__PERPLEXITY, min=5, max=50, value=30),
			input_number_field("Learning rate", id=ids.DR_TSNE__LEARNING_RATE, min=10, max=1000, value=200),
			input_number_field("Number of iterations", id=ids.DR_TSNE__ITERATIONS, min=250, max=50000, value=1000)
		]
	else:
		return [
			input_select_field(
				title="Number of components", 
				id=ids.DR_PCA__COMPONENTS, 
				options=[
					{"label": "2", "value": 2},
					{"label": "3", "value": 3}
	  			]
			)
		]
	
@callback(
	Output(ids.DR__GRAPH, "figure"),

	Input(ids.DR__START_ALGO, "n_clicks"),
	Input(ids.DR__SELECT_ALGO, "value")
)
def start_algo(n_clicks: int, algo: str):
	if algo == DR_ALGO.TSNE:
		pass

def render(app: DashProxy) -> dbc.Card:
	return dbc.Card([
		dbc.CardHeader("Dimension Reduction"),
		dbc.CardBody([
			dcc.Graph(
				id=ids.DR__GRAPH,
				figure={
					"layout": dr_layout
				},
				style={
					"height": "80vh"
				}
			)
		]),
		dbc.CardFooter([
			html.Div([
				input_select_field(
					title="Algorithm", 
					id=ids.DR__SELECT_ALGO,
					options=[
						{"label": "PCA", "value": DR_ALGO.PCA.value},
						{"label": "TSNE", "value": DR_ALGO.TSNE.value}
					]
				),
				html.Div(id=ids.DR__CONFIG)
			]),
			html.Div([
				dbc.Button("Start", id=ids.DR__START_ALGO)
			])
		])
	])