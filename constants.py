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