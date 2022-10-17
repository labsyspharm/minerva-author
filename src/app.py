import numpy
import multiprocessing
from flask import Flask
from waitress import serve

lock_1 = multiprocessing.Lock()
multiprocessing.freeze_support()
app = Flask(__name__)

if __name__ == "__main__":
    print("hello world")
    serve(app, listen="127.0.0.1:8000", threads=10)
