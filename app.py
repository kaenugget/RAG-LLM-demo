from flask import Flask, request
from langchain_community.llms import Ollama

app = Flask(__name__)


# llm = Ollama(model="llama3")

# response = llm.invoke("Tell me a cat joke")

# print(response)


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")


    response_answer = "Sample Response. Query: " + query
    return response_answer




def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()