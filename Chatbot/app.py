"""
app.py — Flask web server for the FAQ Chatbot

Routes:
  GET  /        → serves the chat UI (templates/index.html)
  POST /chat    → accepts {"message": "..."} → returns {"answer": "...", "matched_q": "...", "score": 0.0}
"""

from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__)

# Initialise the chatbot once at startup
bot = FAQChatbot(faqs_path="faqs.json")


@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receive a user message and return the best-matching FAQ answer.

    Request body  (JSON): { "message": "user query here" }
    Response body (JSON): { "answer": "...", "matched_q": "...", "score": 0.0 }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"answer": "Please send a non-empty message.", "matched_q": None, "score": 0.0}), 400

    result = bot.get_answer(user_message)
    return jsonify(result)


if __name__ == "__main__":
    print("Starting FAQ Chatbot server at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
