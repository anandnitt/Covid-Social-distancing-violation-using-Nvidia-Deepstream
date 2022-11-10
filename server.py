
from flask import Flask, jsonify, render_template, request
import webbrowser
import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/_stuff', methods = ['GET'])
def stuff():
    with open('output/jobs0.log', 'r') as f:       
        return jsonify(result=f.read())

@app.route('/_stuff1', methods = ['GET'])
def stuff1():
    with open('output/jobs1.log', 'r') as f:       
        return jsonify(result1=f.read())

@app.route('/_stuff2', methods = ['GET'])
def stuff2():
    with open('output/jobs2.log', 'r') as f:       
        return jsonify(result2=f.read())

@app.route('/_stuff3', methods = ['GET'])
def stuff3():
    with open('output/jobs3.log', 'r') as f:       
        return jsonify(result3=f.read())


@app.route('/')
def index():
    return render_template('dy1.html')


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == "__main__":
	app.run(debug=True,host="127.0.0.1",port=5000)

