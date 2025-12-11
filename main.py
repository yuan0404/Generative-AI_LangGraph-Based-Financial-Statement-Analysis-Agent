from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import init_graph, init_state


app = FastAPI(title="Financial Statement Analysis Agent API")

origins = [
    "http://localhost",
    "http://localhost:5500",
    "https://yuan0404.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Input(BaseModel):
    question: str


load_dotenv()
graph = init_graph()
memory = []


@app.post("/api")
def analyze(input: Input):
    user_input = input.question
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    state = init_state(user_input, memory, time)
    result = graph.invoke(state)

    answer = result.get("final_output")
    memory.append({"user": user_input, "assistant": answer})

    return result

