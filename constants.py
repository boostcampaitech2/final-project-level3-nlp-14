from typing import Optional, Union, Dict
from PIL.Image import Image


from pydantic import BaseModel


class Input(BaseModel):
    query: str
    document: Optional[str]
    image: Optional[Union[Image, str]]

    class Config:
        arbitrary_types_allowed = True


TITLE = "KiYoung2Bot"
DESCRIPTION = """
읽거나 보고, 아는 것만 답변하는 지혜로운 기영이봇
""".strip()

TOKENIZER_USER_AGENT = {
    "file_type": "tokenizer", 
    "from_auto_class": False,
    "is_fast": False,
}

STATIC_FALLBACK_MESSAGES = [
    "무슨 말인지 잘 모르겠어...",
    "내가 알아들을 수 있게 잘 말해줘!",
    "잘 이해가 안되는데 좀 더 자세하게 말해주라 ㅎㅎ",
]

CHATBOT_ANSWER_INPUT = [
    "<answer1>",
    "<answer2>",
    "<answer3>",
    "<answer4>",
]

CHATBOT_NOANSWER_INPUT = [
    "<noanswer1>",
    "<noanswer2>",
    "<noanswer3>",
    "<noanswer4>",
]