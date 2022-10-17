import numpy
from flask import Flask
from waitress import serve

app = Flask(__name__)

if __name__ == "__main__":
    print("hello world")
    serve(app, listen="127.0.0.1:8000", threads=10)
