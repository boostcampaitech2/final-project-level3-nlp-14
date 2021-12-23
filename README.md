# 읽거나📖 보고🔍, 아는 것만 답변하는 지혜로운 기영이봇🤓

## 1. Introduction

### Team KiYOUNG2

_"Korean is all YOU Need for dialoGuE"_

#### 🔅 Members  

김대웅|김채은|김태욱|유영재|이하람|진명훈|허진규|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/41335296?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/60843683?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/47404628?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/53523319?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/35680202?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/37775784?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/88299729?v=4' height=80 width=80px></img>
[Github](https://github.com/KimDaeUng)|[Github](https://github.com/Amber-Chaeeunk)|[Github](https://github.com/taeukkkim)|[Github](https://github.com/uyeongjae)|[Github](https://github.com/hrxorxm)|[Github](https://github.com/jinmang2)|[Github](https://github.com/JeangyuHeo)

#### 🔅 Contribution  

- [`진명훈`](https://github.com/jinmang2) &nbsp; PM • Retro Reader
- [`김대웅`](https://github.com/KimDaeUng) &nbsp; Visual Question Answering
- [`김태욱`](https://github.com/taeukkkim) &nbsp; Open-Domain Question Answering • Dialog
- [`허진규`](https://github.com/JeangyuHeo) &nbsp; Visual Question Answering • Video Question Answering
- [`이하람`](https://github.com/hrxorxm) &nbsp; Frontend • Backend
- [`김채은`](https://github.com/Amber-Chaeeunk) &nbsp; Frontend • Backend
- [`유영재`](https://github.com/uyeongjae) &nbsp; Open-Domain Question Answering • Dialog

## 2. Project Outline

### 프로젝트 목표

* 목적
    * Boostcamp 학습 내용 최종 정리
    * 다양한 형태(텍스트, 이미지)의 입력 정보에 대한 질의응답 서비스 제공
* 주요 기능
    * 일반 상식에 대한 질의응답
    * 사용자 입력 문서, 또는 이미지에 대한 질의응답

### 프로젝트 전체 구조

![project_figure](https://user-images.githubusercontent.com/35680202/147059341-de6f12d3-e9d0-4567-99be-cd46ea46f600.png)

## 3. Demo

### 📖 ODQA 예시
![ODQA 예시](https://user-images.githubusercontent.com/35680202/147240932-0f44c8e1-f55c-417f-a9b3-df48e62eb3d0.gif)

### 👀 VQA 예시
![VQA 예시](https://user-images.githubusercontent.com/35680202/147241018-95e33ffe-da80-434c-a65c-41a8cf820b62.gif)

## 4. How to Use
```
.
├── frontend
│   ├── ...
│   └── dist
├── odqa
│   ├── ...
│   └── inference.py
├── vqa
│   ├── ...
│   └── ban_kvqa.py
├── .gitignore
├── .gitmodules
├── LICENSE
├── README.md
├── app.py
├── constants.py
├── init.sh
├── model.py
├── poetry.lock
├── pyproject.toml
└── version.py
```

아래 명령어로 실행 가능합니다.

```bash
# 프로젝트 다운로드
git clone https://github.com/boostcampaitech2/final-project-level3-nlp-14.git
cd final-project-level3-nlp-14
git submodule update --recursive
# 프론트엔드 환경설정
cd frontend
npm install
npm run build
cd ..
# 백엔드 환경설정
poetry shell
poetry install
poe force-cuda11
poe init-vqa
python app.py
```


## 5. References

### Datasets

- [KorQuAD v2.0](https://korquad.github.io/)
    - 라이센스 : CC BY-ND 2.0 KR
- [KLUE - MRC](https://github.com/KLUE-benchmark/KLUE)
    - 라이센스 : CC BY-SA 4.0
- [KVQA(Korean Visual Question Answering)](https://github.com/SKTBrain/KVQA)
    - 라이센스 : [Korean VQA License](https://github.com/SKTBrain/KVQA/blob/master/LICENSE)
- [AI HUB 개방 데이터](https://aihub.or.kr/aihub-data/natural-language/about)
    - 라이센스 : https://aihub.or.kr/intro/policy


### Paper
- [Antol, Stanislaw, et al. "Vqa: Visual question answering." Proceedings of the IEEE international conference on computer vision. 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf)
- [Yang, Zichao, et al. "Stacked attention networks for image question answering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf)
- [Jin-Hwa Kim, Jaehyun Jun, and Byoung-Tak Zhang. "Bilinear attention networks." Advances in Neural Information Processing Systems 31. 2018](https://papers.nips.cc/paper/2018/file/96ea64f3a1aa2fd00c72faacf0cb8ac9-Paper.pdf)
- [Jin-Hwa Kim, Soohyun Lim, et al. "Korean Localization of Visual Question Answering for Blind People." AI for Social Good workshop at NeurIPS. 2019](https://aiforsocialgood.github.io/neurips2019/accepted/track1/pdfs/44_aisg_neurips2019.pdf)
- [Anderson, Peter, et al. "Bottom-up and top-down attention for image captioning and visual question answering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf)
- [Xu et al., Curriculum Learning for Natural Language Understanding, ACL 2020](https://aclanthology.org/2020.acl-main.542.pdf)
- [ZHANG, Zhuosheng; YANG, Junjie; ZHAO, Hai. Retrospective reader for machine reading comprehension. arXiv preprint arXiv:2001.09694, 2020.](https://arxiv.org/pdf/2001.09694.pdf")

### Software
#### Open-Domain Question Answering - Reader
- [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator)
- [huggingface/datasets](https://github.com/huggingface/datasets)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [retro reader](https://github.com/cooelf/AwesomeMRC)

#### Open-Domain Question Answering - Retrieval
- [elastricsearch](https://github.com/elastic/elasticsearch-py)

#### Visual Question Answering
- [MILVLG/bottom-up-attention](https://github.com/MILVLG/bottom-up-attention.pytorch)
- [Shivanshu-Gupta/Stacked Attention Network](https://github.com/Shivanshu-Gupta/Visual-Question-Answering)
- [SKTBrain/BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)

#### Web Frameworks
- [Vuejs/Vuetify](https://github.com/vuetifyjs/vuetify)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [Stremlit](https://github.com/streamlit/streamlit)
