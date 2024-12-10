from flask import Flask, request, jsonify, render_template
from app.analyzer.pipeline import (
    process_conversation,
    read_conversation_text,
    DEFAULT_MODEL,
)
import asyncio

app = Flask(__name__)


@app.route("/")
def index():
    available_conversations = ["transcript_1"]
    # We'll also provide models to the frontend
    available_models = ["gpt-4o-mini", "ollama:llama3.2:latest", "ollama:qwen2.5:7b"]
    return render_template(
        "index.html", conversations=available_conversations, models=available_models
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    chosen_model = data.get("model_name", DEFAULT_MODEL)
    if not conversation_id:
        return jsonify({"error": "conversation_id not provided"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        process_conversation(conversation_id, chosen_model)
    )

    if "error" in result:
        return jsonify(result), 500

    transcript = read_conversation_text(conversation_id)
    return jsonify({"analysis": result, "transcript": transcript})


if __name__ == "__main__":
    app.run(debug=True)
