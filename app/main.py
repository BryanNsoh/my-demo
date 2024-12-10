# app/main.py
from flask import Flask, request, jsonify, render_template, redirect, url_for
from app.analyzer.pipeline import (
    process_conversation,
    read_conversation_text,
    DEFAULT_MODEL,
    DATA_DIR,
)
import asyncio
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    # Get all txt files in data directory
    transcript_files = [
        f.stem for f in DATA_DIR.glob("*.txt") if not f.stem.endswith("_analysis")
    ]
    transcript_files = sorted(set(transcript_files))

    # For each transcript, check if analysis exists
    transcripts_info = []
    for transcript_id in transcript_files:
        analysis_path = DATA_DIR / f"{transcript_id}_analysis.json"
        has_analysis = analysis_path.exists()
        transcripts_info.append(
            {
                "id": transcript_id,
                "has_analysis": has_analysis,
                "text": read_conversation_text(transcript_id)[:200] + "...",
            }
        )

    return render_template(
        "index.html", models=[DEFAULT_MODEL], transcripts=transcripts_info
    )


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

    return redirect(url_for("results", conversation_id=conversation_id))


@app.route("/results/<conversation_id>")
def results(conversation_id):
    analysis_path = os.path.join(UPLOAD_FOLDER, conversation_id + "_analysis.json")

    # If analysis doesn't exist, create it
    if not os.path.exists(analysis_path):
        # Run the analysis pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            process_conversation(conversation_id, DEFAULT_MODEL)
        )
        loop.close()

        if "error" in result:
            return f"Error during analysis: {result['error']}", 500

        # Save analysis results
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "analysis": result,
                    "transcript": read_conversation_text(conversation_id),
                },
                f,
                indent=2,
            )

    # Read and return analysis results
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
