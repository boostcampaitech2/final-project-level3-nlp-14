import sys
# @TODO Deprecated!
sys.path.append("./vqa")

import pickle
import random
import joblib
from typing import Union, Optional, Dict, Any
from PIL.Image import Image

from odqa.odqa import ODQA
from vqa.ban_kvqa import VQA

from transformers.file_utils import (
    hf_bucket_url,
    cached_path,
)

from constants import STATIC_FALLBACK_MESSAGES, TOKENIZER_USER_AGENT, CHATBOT_ANSWER_INPUT, CHATBOT_NOANSWER_INPUT


class QAManager:
    def __init__(self):
        # Load QA Models for knowledge
        self.vqa = VQA()
        self.odqa = ODQA()
        self._init_chat_module()
        
    def _init_chat_module(self):
        # Download from huggingface hub
        intent_classifier = self.download_from_hf_hub("intent_classifier.pkl") # NLU
        simple_chatbot = self.download_from_hf_hub("chatbot.pkl") # NLG
        answer_candidates = self.download_from_hf_hub("answer_candidates.pkl") # DST
        
        # Set the modules
        self.intent_classifier = self.read_pickle(intent_classifier, use_joblib=True)
        self.simple_chatbot = self.read_pickle(simple_chatbot, use_joblib=True)
        self.answer_candidates = self.read_pickle(answer_candidates, use_joblib=False)
        
    @staticmethod
    def download_from_hf_hub(
        filename: str,
        model_id: str = "kiyoung2/kiyoung2-hub",
        use_auth_token: bool = False,
        user_agent: Dict[str, Any] = TOKENIZER_USER_AGENT,
        force_download: bool = False,
    ) -> str:
        resolved = hf_bucket_url(model_id=model_id, filename=filename)
        file_path = cached_path(
            resolved,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
            force_download=force_download,
        )
        return file_path
    
    @staticmethod
    def read_pickle(file_path: str, use_joblib: bool = False):
        if use_joblib:
            data = joblib.load(file_path)
        else:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        return data
        
    def get_embed_feature(self, query):
        # Tokenizing
        input_ids = self.odqa.tokenizer(query, return_tensors="pt")["input_ids"]
        # Get word embedding features (with numpify)
        word_embedding = self.odqa.word_embeddings(input_ids)
        feature = word_embedding.mean(dim=1).view(-1).detach().numpy()
        return feature
        
    def identify_intent(self, query):
        feature = self.get_embed_feature(query)
        # Determining what the intent of the question is
        qa_ratio = self.intent_classifier.predict_proba([feature])[0][0]
        return qa_ratio # Question that requires knowledge ~ (0, 1)

    def answer(
        self,
        query: str,
        document: Optional[str] = None,
        image: Optional[Union[str, Image]] = None,
    ):
        if image is not None:
            answer, answerable = self.vqa.answer(query, image)
        elif document is not None:
            answer, answerable = self.odqa.answer(query, document)
        else:
            answer, answerable = "", False
        
        return answer, answerable
    
    def chat(self, query: str, thresh: Optional[float] = None):
        if query == "<answer>":
            ind = random.randint(0, len(CHATBOT_ANSWER_INPUT)-1)
            query = CHATBOT_ANSWER_INPUT[ind]
        elif query == "<noanswer>":
            ind = random.randint(0, len(CHATBOT_NOANSWER_INPUT)-1)
            query = CHATBOT_NOANSWER_INPUT[ind]

        feature = self.get_embed_feature(query)
        probs = self.simple_chatbot.predict_proba([feature])[0]
        max_index = probs.argmax()
        if thresh and probs[max_index] < thresh:
            ind = random.randint(0, len(STATIC_FALLBACK_MESSAGES)-1)
            response = STATIC_FALLBACK_MESSAGES[ind]
        else:
            response = self.answer_candidates[max_index]
        return response