import sys
# @TODO Deprecated!
# sys.path.append("./vqa")

from typing import Union, Optional
from PIL.Image import Image

from odqa.odqa.odqa import ODQA
# from vqa.ban_kvqa import VQA


class QAManager:
    def __init__(self):
        # self.vqa = VQA()
        self.odqa = ODQA()

    def __call__(
        self,
        query: str,
        document: Optional[str] = None,
        image: Optional[Union[str, Image]] = None,
    ):
        if image is not None:
            answer, answerable = self.vqa.answer(query, image)
            answer = ""
            answerable = False
            print("이미지를 잘 받아왔습니다.", image)
        else:
            answer, answerable = self.odqa.answer(query, document)
        return answer, answerable