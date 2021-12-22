#!bin/bash
cd vqa/bottom_up_attention_pytorch/detectron2
pip install -e . -I
cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
python setup.py build develop

wget https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1 -O bua-caffe-frcn-r101_with_attributes.pth

cd ..
bash download.sh