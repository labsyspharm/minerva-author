import sys
import os
from flask import Flask
from flask import jsonify


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder at _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


app = Flask(__name__,
            static_folder=resource_path('static'),
            static_url_path='')

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/image')
def hello():
    return jsonify({"hello": "world"})

if __name__ == '__main__':
    app.run(debug=True, port=2020)
