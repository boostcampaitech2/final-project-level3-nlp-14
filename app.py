import re
# import sys
# sys.path.append('./vqa/bottom_up_attention_pytorch/detectron2')
# sys.path.append('./vqa/bottom_up_attention_pytorch/')

import logging

from PIL import Image
from typing import Optional
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from model import QAManager
from version import VERSION
from constants import (
    Union, Dict,
    Input,
    TITLE, DESCRIPTION
)


# Set logger
logging.basicConfig(filename="app.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.info("Running kiyoung2 app")
logger = logging.getLogger(__name__)

# Set up FastAPI
app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION,
)
api = FastAPI(openapi_prefix="/api")
app.mount("/api", api)
app.mount('/', StaticFiles(directory='frontend/dist', html=True), name="static")

# Set up Question Answering Manager
qa_manager = QAManager()

# @Deprecated: Set threshold
thresh: float = 0.5


@api.post("/chat")
async def chat(
    query: str = Form(...),
    document: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
) -> Dict[str, str]:
    """ """
    # Preprocess query
    query = re.sub("\?", "", query)
    # Process Context
    if image is not None and image.filename != "":
        contents = await image.read()
        with open(f"tmp/{image.filename}", "wb") as f:
            f.write(contents)
        img = Image.open(f"tmp/{image.filename}")
    else:
        img = None
    
    # Identify whether knowledge is needed
    qa_ratio = qa_manager.identify_intent(query)
    
    if (qa_ratio >= thresh) or (image is not None and image.filename != "") or (document is not None):
        answer, answerable = qa_manager.answer(query, document, img)
        chatbot_input = "<answer>" if answerable else "<noanswer>"
        response = qa_manager.chat(chatbot_input)
        answers = [answer, response]
    else:
        response = qa_manager.chat(query, thresh=thresh)
        answers = [response]
    logging.info(f"query: {query}\ttype(doc){type(document)}\ttype(img){type(img)}\t"
                 f"qa_ratio: {qa_ratio}\tthresh: {thresh}\tanswers: {answers}")
    return answers


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)