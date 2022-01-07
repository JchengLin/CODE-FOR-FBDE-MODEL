
# This repository provides the official Tensorflow implementation of the following paper:
FBDE: Full-body De-identification via Adversarial Learning and Contrastive Learning

Jiacheng Lin, Zhiqiang Xiao, Yaping Li, Xiewen Dai, Zhiyong Li, and Shutao Li.

ABSTRCT-Visual privacy protection has become difficult because of the large-scale application of visual devices. Although the methods of visual privacy protection have developed rapidly, it is mostly in face de-identification or visual privacy encryption. At present, there is still a lack of a method for full-body de-identification, especially using deep learning methods. Based on this, we propose a full-body de-identification model based on adversarial learning and contrastive learning (FBDE). Firstly, the architecture of the generator and discriminator is designed for full-body de-identification. Secondly, a content mapping network and content loss function based on contrastive learning is designed for full-body de-identification. Then, to address the problem of misjudgment of discriminators in the process of full-body de-identification, an adversarial loss based on triple loss is proposed. Furthermore, a full-body deidentification dataset for full-body de-identification is made and implemented. Finally, the experiment results show that the FBDE model is not only better than state-of-the-art full-body de-identification models of parameters and training speed, but also in the effect of full-body de-identification. In addition, to the best of our knowledge, this is the first-time framework used to perform full-body de-identification.

KEYWORD-Full-body De-identification, Data privacy, Adversarial learning, Contrastive learning, Neural networks.

# Requirements

python == 3.6

tensorflow == 1.14


# Dataset

You can download this dataset on: https://github.com/JchengLin/FBDE_DATASET

# Train

RUN python main.py --dataset selfie2anime

# Test

RUN python main.py --dataset selfie2anime --phase test

# Evaluation

The evaluation protocal in the EVALUATION PROTOCAL.
