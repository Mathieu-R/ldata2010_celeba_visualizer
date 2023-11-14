from dash_extensions.enrich import DashProxy, html, dcc
from .components import header, navbar

from ui import ids

def createLayout(app: DashProxy):
	return html.Div(
		children=[
			dcc.Loading([
				dcc.Store(id=ids.FEATURES_STORE, storage_type="memory"),
				dcc.Store(id=ids.IMAGES_STORE, storage_type="memory"),
				dcc.Store(id=ids.EMBEDDINGS_STORE, storage_type="memory"),
			], fullscreen=True, loading_state={"type": "circle"}),
			header.render(app),
			navbar.render(app)
		],
		className="app"
	)