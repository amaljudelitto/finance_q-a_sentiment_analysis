from fastapi import FastAPI, Request
from pydantic import BaseModel
from infer import infer

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Finance LLM is running!"}

@app.post("/generate/")
def generate_answer(request: PromptRequest):
    response = infer(request.prompt)
    return {"response": response}
