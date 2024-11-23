from flask import Flask, request, jsonify, render_template
from chains import rag_chain
from history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)

# Initialize the conversational RAG chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("input", "")
    session_id = data.get("session_id", "default_session")

    # Get the response from the chain
    result = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(debug=True)