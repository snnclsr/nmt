from datetime import datetime

from config import HOST, PORT, DEBUG
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/translate", methods=["GET", "POST"])
def translate():
    
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        args = request.form

    print(args)
    text_input = args["textarea"]
    print("Input: ", text_input)

    result_hypothesis = [
        "123",
        "345",
        "678"
    ]
    return render_template("index.html", hypothesis=result_hypothesis)



if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)