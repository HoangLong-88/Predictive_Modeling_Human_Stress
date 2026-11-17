from flask import Flask, render_template, request
from pathlib import Path
import joblib

app = Flask(__name__)

# 1. Load trained ML model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "stress_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except (OSError, ValueError):
    app.logger.exception("Model could not be loaded")
    model = None

# 2. Map prediction labels
LABEL_NAMES = {
    0: "Fewer stress-related signals",
    1: "Stress-related signals detected"
}


# 3. Define website route
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    text = ""
    error = None

    if request.method == "POST":

        # Get user input
        text = request.form.get("text", "").strip()

        if not text:
            error = "Please enter some text."

        elif len(text) > 5000:
            error = "Text must contain at most 5000 characters."

        elif model is None:
            error = "The model is temporarily unavailable. Please try again later."

        elif not any(character.isalpha() for character in text):
            error = "Please include some words so the model can analyze your text."

        else:
            # Make prediction
            prediction = int(model.predict([text])[0])

            result = {
                "label": LABEL_NAMES[prediction],
                "stress_probability": None
            }

            # Get model probability if supported
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba([text])[0]

                stress_index = list(model.classes_).index(1)

                result["stress_probability"] = round(
                    float(probabilities[stress_index]) * 100,
                    2
                )

    # 4. Return results to frontend
    return render_template(
        "index.html",
        result=result,
        text=text,
        error=error
    )


if __name__ == "__main__":
    app.run()
