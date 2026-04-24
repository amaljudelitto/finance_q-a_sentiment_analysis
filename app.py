from fastapi import FastAPI
from pydantic import BaseModel
from infer import generate_answer

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "qa"  # Defaults to Q&A if the user doesn't specify

@app.get("/")
def read_root():
    return {"message": "Finance LLM API is running!"}

@app.post("/generate/")
def generate(request: PromptRequest):
    # Pass both the prompt and the task type to your infer script
    response = generate_answer(request.prompt, task_type=request.task_type)
    return {"response": response}
