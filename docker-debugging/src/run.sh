#!/bin/bash
# run.sh
# Script for starting up the Flask app
echo "Starting Flask App..."

export FLASK_APP=app.py

flask run --host=0.0.0.0 --port=5000
