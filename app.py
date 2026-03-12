from flask import Flask, render_template, request
import whisper
import os
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Models
whisper_model = whisper.load_model("base")
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    sentiment = ""

    if request.method == "POST":

        if "audio" not in request.files:
            return "No file uploaded"

        file = request.files["audio"]

        if file.filename == "":
            return "No file selected"

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Transcription
        result = whisper_model.transcribe(filepath)
        transcription = result["text"]

        # Sentiment Analysis
        sentiment_result = sentiment_pipeline(transcription)
        sentiment = sentiment_result[0]["label"]

    return render_template(
        "index.html",
        transcription=transcription,
        sentiment=sentiment
    )


if __name__ == "__main__":
    app.run(debug=True)