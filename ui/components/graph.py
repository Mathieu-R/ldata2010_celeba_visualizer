from dash_extensions.enrich import dcc
from plotly import graph_objs as go

def custom_graph(id: str, no_margin: bool = False):
	if (no_margin):
		layout = go.Layout(
			margin=dict(l=0, r=0, b=0, t=0)
		)
	else: 
		layout = go.Layout(
			margin=dict(l=5, r=5, b=5, t=5)
		)
	return dcc.Graph(
		id=id, 
		figure={
			"layout": layout
		},
		config=dict(
			modeBarButtons=[["toImage"]], displaylogo=False, displayModeBar=False
		)
	)