from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="/")

@app.route("/")
def index() :
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict() :
    columns = list(request.form)
    values = []
    for item in request.form :
        values.append(request.form[item])
    
    df = pd.DataFrame([values], columns=columns)
    median_house_value = model.predict(pipeline.transform(df))

    return render_template("result.html", median_house_value=f"{median_house_value[0]:.2f}")

if __name__ == "__main__" :
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    app.run(debug=True)