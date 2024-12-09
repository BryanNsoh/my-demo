from flask import Flask, request, jsonify, render_template
from app.analyzer.pipeline import process_conversation, read_conversation_text
import asyncio

app = Flask(__name__)


@app.route("/")
def index():
    available_conversations = ["transcript_1"]
    return render_template("index.html", conversations=available_conversations)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id not provided"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_conversation(conversation_id))

    if "error" in result:
        return jsonify(result), 500

    transcript = read_conversation_text(conversation_id)
    return jsonify({"analysis": result, "transcript": transcript})


if __name__ == "__main__":
    app.run(debug=True)
