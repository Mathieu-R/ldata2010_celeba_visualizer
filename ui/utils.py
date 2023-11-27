import dash_bootstrap_components as dbc
from dash_extensions.enrich import html

def make_hideable(element, hide: bool =False):
    """helper function to optionally not display an element in a layout.

    This is used for all the hide_ flags in ExplainerComponent constructors.
    e.g. hide_cutoff=True to hide a cutoff slider from a layout:

    Example:
        make_hideable(dbc.Col([cutoff.layout()]), hide=hide_cutoff)

    Args:
        hide(bool): wrap the element inside a hidden html.div. If the element
                    is a dbc.Col or a dbc.Row, wrap element.children in
                    a hidden html.Div instead. Defaults to False.
    """
    if hide:
        if isinstance(element, dbc.Col) or isinstance(element, dbc.Row):
            return html.Div(element.children, style=dict(display="none"))
        else:
            return html.Div(element, style=dict(display="none"))
    else:
        return element