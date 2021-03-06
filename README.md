# μ½κ±°λπ λ³΄κ³ π, μλ κ²λ§ λ΅λ³νλ μ§νλ‘μ΄ κΈ°μμ΄λ΄π€

## 1. Introduction

### Team KiYOUNG2

_"Korean is all YOU Need for dialoGuE"_

#### π Members  

κΉλμ|κΉμ±μ|κΉνμ±|μ μμ¬|μ΄νλ|μ§λͺν|νμ§κ·|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/41335296?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/60843683?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/47404628?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/53523319?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/35680202?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/37775784?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/88299729?v=4' height=80 width=80px></img>
[Github](https://github.com/KimDaeUng)|[Github](https://github.com/Amber-Chaeeunk)|[Github](https://github.com/taeukkkim)|[Github](https://github.com/uyeongjae)|[Github](https://github.com/hrxorxm)|[Github](https://github.com/jinmang2)|[Github](https://github.com/JeangyuHeo)

#### π Contribution  

- [`μ§λͺν`](https://github.com/jinmang2) &nbsp; PM β’ Retro Reader
- [`κΉλμ`](https://github.com/KimDaeUng) &nbsp; Visual Question Answering
- [`κΉνμ±`](https://github.com/taeukkkim) &nbsp; Open-Domain Question Answering β’ Dialog
- [`νμ§κ·`](https://github.com/JeangyuHeo) &nbsp; Visual Question Answering β’ Video Question Answering
- [`μ΄νλ`](https://github.com/hrxorxm) &nbsp; Frontend β’ Backend
- [`κΉμ±μ`](https://github.com/Amber-Chaeeunk) &nbsp; Frontend β’ Backend
- [`μ μμ¬`](https://github.com/uyeongjae) &nbsp; Open-Domain Question Answering β’ Dialog

## 2. Project Outline

### νλ‘μ νΈ λͺ©ν

* λͺ©μ 
    * Boostcamp νμ΅ λ΄μ© μ΅μ’ μ λ¦¬
    * λ€μν νν(νμ€νΈ, μ΄λ―Έμ§)μ μλ ₯ μ λ³΄μ λν μ§μμλ΅ μλΉμ€ μ κ³΅
* μ£Όμ κΈ°λ₯
    * μΌλ° μμμ λν μ§μμλ΅
    * μ¬μ©μ μλ ₯ λ¬Έμ, λλ μ΄λ―Έμ§μ λν μ§μμλ΅

### νλ‘μ νΈ μ μ²΄ κ΅¬μ‘°

![project_figure](https://user-images.githubusercontent.com/35680202/147059341-de6f12d3-e9d0-4567-99be-cd46ea46f600.png)

## 3. Demo

### π ODQA μμ
![ODQA μμ](https://user-images.githubusercontent.com/35680202/147240932-0f44c8e1-f55c-417f-a9b3-df48e62eb3d0.gif)

### π VQA μμ
![VQA μμ](https://user-images.githubusercontent.com/35680202/147241018-95e33ffe-da80-434c-a65c-41a8cf820b62.gif)

## 4. How to Use
```
.
βββ frontend
β   βββ ...
β   βββ dist
βββ odqa
β   βββ ...
β   βββ inference.py
βββ vqa
β   βββ ...
β   βββ ban_kvqa.py
βββ .gitignore
βββ .gitmodules
βββ LICENSE
βββ README.md
βββ app.py
βββ constants.py
βββ init.sh
βββ model.py
βββ poetry.lock
βββ pyproject.toml
βββ version.py
```

μλ λͺλ Ήμ΄λ‘ μ€ν κ°λ₯ν©λλ€.

```bash
# νλ‘μ νΈ λ€μ΄λ‘λ
git clone https://github.com/boostcampaitech2/final-project-level3-nlp-14.git --recursive
cd final-project-level3-nlp-14
git submodule update --recursive
# νλ‘ νΈμλ νκ²½μ€μ 
cd frontend
npm install
npm run build
cd ..
# λ°±μλ νκ²½μ€μ 
poetry shell
poetry install
poe force-cuda11
poe init-vqa
python app.py
```


## 5. References

### Datasets

- [KorQuAD v2.0](https://korquad.github.io/)
    - λΌμ΄μΌμ€ : CC BY-ND 2.0 KR
- [KLUE - MRC](https://github.com/KLUE-benchmark/KLUE)
    - λΌμ΄μΌμ€ : CC BY-SA 4.0
- [KVQA(Korean Visual Question Answering)](https://github.com/SKTBrain/KVQA)
    - λΌμ΄μΌμ€ : [Korean VQA License](https://github.com/SKTBrain/KVQA/blob/master/LICENSE)
- [AI HUB κ°λ°© λ°μ΄ν°](https://aihub.or.kr/aihub-data/natural-language/about)
    - λΌμ΄μΌμ€ : https://aihub.or.kr/intro/policy


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
