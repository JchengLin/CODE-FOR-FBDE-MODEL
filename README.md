# Code-for-FBDE-model
This is a repository for "FBDE: Full-body De-identification via Adversarial Learning and Contrastive Learning."

# This repository provides the official Tensorflow implementation of the following paper:
FBDE: Full-body De-identification via Adversarial Learning and Contrastive Learning

Jiacheng Lin, Zhiqiang Xiao, Yaping Li, Xiewen Dai, Zhiyong Li, Member, IEEE, and Shutao Li, Fellow IEEE.

ABSTRCT-Visual privacy protection has become difficult because of the large-scale application of visual devices. Although the methods of visual privacy protection have developed rapidly, it is mostly in face de-identification or visual privacy encryption. At present, there is still a lack of a method for full-body de-identification, especially using deep learning methods. Based on this, we propose a full-body de-identification model based on adversarial learning and contrastive learning (FBDE). Firstly, the architecture of the generator and discriminator is designed for full-body de-identification. Secondly, a content mapping network and content loss function based on contrastive learning is designed for full-body de-identification. Then, to address the problem of misjudgment of discriminators in the process of full-body de-identification, an adversarial loss based on triple loss is proposed. Furthermore, a full-body deidentification dataset for full-body de-identification is made and implemented. Finally, the experiment results show that the FBDE model is not only better than state-of-the-art full-body de-identification models of parameters and training speed, but also in the effect of full-body de-identification. In addition, to the best of our knowledge, this is the first-time framework used to perform full-body de-identification.

KEYWORD-Full-body De-identification, Data privacy, Adversarial learning, Contrastive learning, Neural networks.

# Requirements

python == 3.6

tensorflow == 1.14


# DATASET

FBDE LEVEL I:   The image’s characters, dress styles, and backgrounds in trainA and trainB are unified. The images numbers of trainA, trainB, testA is 1050, 1050, and 100, respectively.

FBDE LEVEL II:  The images in trainA and trainB have the same background, different dressing styles, and characters. The images numbers of trainA, trainB, testA is 1230, 1230, and 100, respectively.

FBDE LEVEL III: The dress styles of trainA and trainB are uniform, the characters and backgrounds are different. The images numbers of trainA, trainB, testA is 1050, 1050, and 100, respectively.

FBDE LEVEL IV:  The characters, dress styles, and backgrounds in trainA and trainB are not uniform. The images numbers of trainA, trainB, testA is 2780, 2770, and 138, respectively.

You can download this dataset on: https://github.com/JchengLin/FBDE_DATASET

# Usage

├── dataset

   └── YOUR_DATASET_NAME   
   
       ├── trainA
       
           ├── xxx.jpg (name, format doesn't matter)
           
           ├── yyy.png
           
           └── ...
           
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...

# TRAIN

RUN python main.py --dataset selfie2anime

# TEST

RUN python main.py --dataset selfie2anime --phase test

# EVALUATION PROTOCAL

The evaluation protocal in the EVALUATION PROTOCAL.
