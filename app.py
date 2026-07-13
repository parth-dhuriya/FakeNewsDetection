from flask import Flask, render_template, request

from models import manual_testing

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    result = manual_testing(news)

    return render_template(
        "index.html",
        result=result,
        news=news
    )

if __name__ == "__main__":
    app.run(debug=True)
    