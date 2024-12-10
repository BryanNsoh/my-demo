from flask import Flask, request, jsonify, render_template, redirect, url_for
from app.analyzer.pipeline import (
    process_conversation,
    read_conversation_text,
    DEFAULT_MODEL,
)
import asyncio
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    # Landing page: user chooses conversation_id, model, inputs either text or audio
    return render_template("index.html", models=[DEFAULT_MODEL])


@app.route("/upload_and_analyze", methods=["POST"])
def upload_and_analyze():
    conversation_id = request.form.get("conversation_id", "").strip()
    if not conversation_id:
        return "Please provide a conversation ID", 400

    model_name = request.form.get("model_name", DEFAULT_MODEL).strip()
    input_type = request.form.get("input_type")  # "text" or "audio"

    if input_type == "text":
        text = request.form.get("text_input", "").strip()
        if not text:
            return "Please provide transcript text", 400
        # Save text
        text_path = os.path.join(UPLOAD_FOLDER, conversation_id + ".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
    elif input_type == "audio":
        file = request.files.get("audio_file")
        if not file:
            return "Please upload an audio file", 400
        audio_path = os.path.join(UPLOAD_FOLDER, conversation_id + ".mp3")
        file.save(audio_path)
    else:
        return "Invalid input type", 400

    # Run the analysis pipeline
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_conversation(conversation_id, model_name))

    if "error" in result:
        return f"Error: {result['error']}", 500

    # Save analysis results to a JSON file for retrieval by /results/<conversation_id>
    analysis_path = os.path.join(UPLOAD_FOLDER, conversation_id + "_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(
            {"analysis": result, "transcript": read_conversation_text(conversation_id)},
            f,
            indent=2,
        )

    return redirect(url_for("results", conversation_id=conversation_id))


@app.route("/results/<conversation_id>")
def results(conversation_id):
    analysis_path = os.path.join(UPLOAD_FOLDER, conversation_id + "_analysis.json")
    if not os.path.exists(analysis_path):
        return f"No analysis found for {conversation_id}", 404

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    analysis = data["analysis"]
    transcript = data["transcript"]

    return render_template(
        "results.html",
        analysis=analysis,
        transcript=transcript,
        conversation_id=conversation_id,
    )


if __name__ == "__main__":
    app.run(debug=True)
