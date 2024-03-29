from pathlib import Path
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
from whisper.transcribe import transcribe
import whisper

UPLOAD_FOLDER = Path("/tmp/audio/")
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

ALLOWED_EXTENSIONS = {"flac", "wav", "mp3", "ogg"}
model = None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = app.config["UPLOAD_FOLDER"] / filename
            file.save(str(filepath))
            return get_results(filepath)
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


def get_results(filename):
    global model
    if model is None:
        model = whisper.load_model("small")

    result = transcribe(model=model, audio=str(filename), word_timestamps=True)
    result = result or {}

    text = ""
    for s in result.get("segments", []):
        text += f"[{s['start']:0.1f} -> {s['end']:0.1f}] {s['text']}<br>"

    # print the recognized text
    return f"""
    <!doctype html>
    <title>Upload new File</title>
    language: { result.get("language") }
    <br>
    <hr>
    <br>
    { text }
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
