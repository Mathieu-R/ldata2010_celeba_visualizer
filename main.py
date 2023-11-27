import dash_bootstrap_components as dbc

from dash_extensions.enrich import DashProxy, ServersideOutputTransform 

from constants import DEBUG
from ui.app import createLayout

def main() -> DashProxy:
	# keep data on server, avoiding transfer between client and server
	# therefore, we don't need performing JSON serialization (that is really slow...)
	app = DashProxy(__name__, transforms=[ServersideOutputTransform()], external_stylesheets=[dbc.themes.SLATE], prevent_initial_callbacks=False)
	app.title = "CelebA Visualization"
	app.layout = createLayout(app)
	return app

if __name__ == "__main__":
	app = main()
	app.run(debug=DEBUG)
	