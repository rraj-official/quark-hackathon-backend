# app.py
import io
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import uvicorn
from pydub import AudioSegment
from pydub.playback import play
# Import our language chain components and helper functions
from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from models import check_if_model_is_available
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain.prompts.prompt import PromptTemplate

# Import our prompt templates and helper functions from llm.py
from llm import CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, _combine_documents, memory

# Import the speech-to-text function (assumed to be implemented in speech.py)
from speech import speech_to_text

app = FastAPI()

# Configure CORS as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response data models
class ChatMessage(BaseModel):
    content: str
    role: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class AnswerResponse(BaseModel):
    answer: str

def translate_text(text, source='auto', target='es'):
    url = 'http://localhost:5000/translate'
    payload = {
        'q': text,
        'source': source,
        'target': target
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=payload, headers=headers)
    if response.ok:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

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
    # Remove streaming callbacks so that we can capture the answer
    answer_chain = {
        "answer": final_inputs | ANSWER_PROMPT | llm.with_config(callbacks=[]),
        "docs": itemgetter("docs"),
    }
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer_chain

    # Define a function that runs the chain and returns the answer.
    def chat_api(question: str) -> str:
        inputs = {"question": question}
        result = final_chain.invoke(inputs)
        # Optionally save to memory
        memory.save_context(inputs, {"answer": result["answer"]})
        return result["answer"]

    # Save our callable chain on the FastAPI app state.
    app.state.chat = chat_api

# Endpoint to process text input.
@app.post("/text", response_model=AnswerResponse)
def process_text(request: ChatRequest):
    try:
        # Process the content of the last message in the list.
        text = request.messages[-1].content
        translated_text_json_1 = translate_text(text, target='en')
        answer = app.state.chat(translated_text_json_1['translatedText'])
        translated_text_json_2 = translate_text(answer, source='en', target = translated_text_json_1['detectedSourceLanguage'])
        return AnswerResponse(answer=translated_text_json_2['translatedText'])
        # answer = app.state.chat(text)
        # return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to process voice input.
@app.post("/voice", response_model=AnswerResponse)
async def process_voice(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        # file_bytes = await file.read()
        
        # # Wrap the bytes in a BytesIO object
        # audio_file = io.BytesIO(file_bytes)
        
        # # Determine the file format from the filename (e.g., 'wav', 'mp3')
        # # You might want to add more robust checking depending on your needs.
        # file_extension = file.filename.split(".")[-1]
        
        # # # Load the audio file using PyDub
        # audio_segment = AudioSegment.from_file(audio_file, format=file_extension)
        
        # # # Play the audio file
        # play(audio_segment)
        
        # # If your speech_to_text function expects a file-like object,
        # # reset the pointer in the BytesIO object.
        # audio_file.seek(0)
        
        # # Convert the uploaded audio file to text
        text_input, lang = speech_to_text(file)
        print(text_input)
        # # Pass the transcribed text to the chat chain
        if lang != 'en':
            translated_text_json = translate_text(text_input, lang, 'en')
            answer = app.state.chat(translated_text_json['translatedText'])
            translated_text_json = translate_text(answer, 'en', lang)
            return AnswerResponse(answer=translated_text_json['translatedText'])
        else:
            answer = app.state.chat(text_input)
            return AnswerResponse(answer=answer)
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    # Run the app with uvicorn. The reload flag is handy for development.
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
