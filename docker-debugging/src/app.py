"""app.py

Simple Flask app with a single endpoint at /.
"""
from flask import Flask

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"
