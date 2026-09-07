"""WSGI entry point for Gunicorn (``gunicorn wsgi:application``).

app.py builds the Dash app and exposes the underlying Flask instance as
``server``; ``application`` is the name the Dockerfile CMD refers to.
"""

from app import server as application

__all__ = ["application"]
