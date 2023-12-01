import os
import dash_bootstrap_components as dbc

from dash import CeleryManager, DiskcacheManager
from dash_extensions.enrich import DashProxy, ServersideOutputTransform

from constants import DEBUG
from ui.app import createLayout

if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

def main() -> DashProxy:
	# keep data on server, avoiding transfer between client and server
	# therefore, we don't need performing JSON serialization (that is really slow...)
	app = DashProxy(__name__, transforms=[ServersideOutputTransform()], external_stylesheets=[dbc.themes.SLATE], background_callback_manager=background_callback_manager,  prevent_initial_callbacks=False)
	app.title = "CelebA Visualization"
	app.layout = createLayout(app)
	return app

if __name__ == "__main__":
	app = main()
	app.run(debug=DEBUG)
	