import sys
# @TODO Deprecated!
sys.path.append("./vqa")

from typing import Union, Optional
from PIL.Image import Image

from odqa.odqa.odqa import ODQA
from vqa.ban_kvqa import VQA


class QAManager:
    def __init__(self):
        self.vqa = VQA()
        self.odqa = ODQA()

    def __call__(
        self,
        query: str,
        document: Optional[str] = None,
        image: Optional[Union[str, Image]] = None,
    ):
        assert document is not None and image is not None, (
            ""
        )
        if image is not None:
            answer, answerable = self.vqa.answer(query, image)
        else:
            answer, answerable = self.odqa.answer(query, document)
        return answer, answerable