# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import uvicorn

from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from models import check_if_model_is_available

# Import some functions and classes used to build the chain
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain.prompts.prompt import PromptTemplate

# Import our prompt templates and helper functions from llm.py
from llm import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, _combine_documents, memory

app = FastAPI()

# Define a model for each individual message
class Message(BaseModel):
    role: str
    content: str

# Update the request model to accept an array of messages
class QuestionRequest(BaseModel):
    messages: List[Message]

# Response model remains the same
class AnswerResponse(BaseModel):
    answer: str

# On startup, we load our models and build our chain.
@app.on_event("startup")
def startup_event():
    # Use defaults; adjust or wire in CLI args as needed.
    llm_model_name = "mistral"
    embedding_model_name = "all-minilm"
    documents_path = "Research"

    # Check if the models are available locally (or pull them if not)
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print("Error with models:", e)
        sys.exit(1)

    # Build (or load) the vector database from your documents
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Create the LLM instance using Ollama
    llm = Ollama(model=llm_model_name)

    # Build the retrieval chain.
    # (This is adapted from your getChatChain but modified so that the function returns the answer.)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    # Here we remove streaming callbacks so that we can capture the answer
    answer_chain = {
        "answer": final_inputs | ANSWER_PROMPT | llm.with_config(callbacks=[]),
        "docs": itemgetter("docs"),
    }
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer_chain

    # Define a function that runs the chain and returns the answer.
    def chat_api(question: str) -> str:
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        # Save to memory (if desired)
        memory.save_context(inputs, {"answer": result["answer"]})
        return result["answer"]

    # Save our callable chain on the FastAPI app state.
    app.state.chat = chat_api

# Define the API endpoint that accepts a messages array and returns an answer.
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")
        # Use the content of the last message as the question
        question = request.messages[-1].content
        answer = app.state.chat(question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the app with uvicorn. The reload flag is handy for development.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
