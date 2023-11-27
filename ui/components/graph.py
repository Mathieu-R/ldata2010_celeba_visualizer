from dash_extensions.enrich import dcc

def custom_graph(id: str):
	return dcc.Graph(
		id=id, 
		config=dict(
			modeBarButtons=[["toImage"]], displaylogo=False, displayModeBar=False
		)
	)