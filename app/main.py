from flask import Flask, request, jsonify, render_template
from app.analyzer.pipeline import process_conversation

app = Flask(__name__)


@app.route("/")
def index():
    available_conversations = ["transcript_1", "transcript_2", "transcript_3"]
    return render_template("index.html", conversations=available_conversations)


@app.route("/analyze", methods=["POST"])
async def analyze():
    data = request.get_json()
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id not provided"}), 400
    result = await process_conversation(conversation_id)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
