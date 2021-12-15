import sys
sys.path.append('./vqa/bottom_up_attention_pytorch/detectron2')
sys.path.append('./vqa/bottom_up_attention_pytorch/')

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import QAManager
from version import VERSION
from constants import (
    Union, Dict,
    Input,
    TITLE, DESCRIPTION
)


app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION,
)
qa_manager = QAManager()
origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
def chat(
    input: Input,
) -> Dict[str, str]:
    query = input.query
    document = input.document
    image = input.image
    answer, answerable = qa_manager(query, document, image)
    if not answerable:
        answer = "질문에 답하기가 어려워요 ㅠㅠ"
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)