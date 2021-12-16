import sys
sys.path.append('./vqa/bottom_up_attention_pytorch/detectron2')
sys.path.append('./vqa/bottom_up_attention_pytorch/')

from PIL import Image
from typing import Optional
import uvicorn
from fastapi import FastAPI, Form, UploadFile
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
async def chat(
    query: str = Form(...),
    document: Optional[str] = Form(None),
    image: Optional[UploadFile] = Form(None),
) -> Dict[str, str]:
    if image is not None and image.filename != "":
        contents = await image.read()
        with open(f"tmp/{image.filename}", "wb") as f:
            f.write(contents)
        img = Image.open(f"tmp/{image.filename}")
    else:
        img = None
    
    answer, answerable = qa_manager(query, document, img)
    if not answerable:
        answer = "질문에 답하기가 어려워요 ㅠㅠ"
    answers = [answer]
    return answers


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)