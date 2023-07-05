# Representation Learning MSc course Summer Semester 2023

## Summary
This course is tailored for MSc students of the AI and Data Science Master of the Heinrich Heine University of Dusseldorf. 

We provide all the course materials, including lectures, slides, and exercise classes. 

[YouTube Playlist of videos](https://www.youtube.com/playlist?list=PL3mKiGE4zNJJ83K4c3IBka6eYfe6v71dS)


## Week 1 - Introduction to Representation Learning
- [Lecture](https://www.youtube.com/watch?v=i1-OtPa9doY&list=PL3mKiGE4zNJJ83K4c3IBka6eYfe6v71dS&index=2) || [Slides](https://uni-duesseldorf.sciebo.de/s/2h3pY73kHHIWtUW)
- Introduction to autoencoders for representation learning, early/old traditional approaches. Based on [Bengio et al. 2012 paper](https://arxiv.org/pdf/1206.5538.pdf)

#### Exercise
Image autoencoders. Learning to use and evaluate the intermediate learned representation.


## Week 2 - Overview of visual self-supervised learning methods
- [Lecture](https://youtu.be/3Zvo1BihTRE) || [Slides](https://uni-duesseldorf.sciebo.de/s/J5f839uJRKQhW8y)
- Self-supervised learning VS Transfer Learning. Pretext VS Downstream Task
- Pretext tasks covered: Colorization, Jigsaw puzzles, Image inpainting, Shuffle and Learn (Videos), - Classify corrupted images, Rotation Prediction
- Semi-supervised learning: Consistency loss
- A small intro to Contrastive loss (infoNCE)

#### Exercise
In this exercise, we will train a ResNet18 on the task of rotation prediction. Rotation prediction provides a simple, yet effective way to learn rich representations from unlabeled image data. The basic idea behind rotation prediction is that the network is trained to predict the orientation of a given image after it has been rotated by a certain angle (e.g., 0°, 90°, 180°, or 270°).


##  Week 3 - BERT:Learning Natural Language Representations
- [Lecture](https://youtu.be/qCZiR5I47Bo) || [Slides](https://uni-duesseldorf.sciebo.de/s/a3XfMv2HgcxMR8a)
- Natural Language Processing (NLP) basics
- RNN, self-attention, and Transformer recap
- Language pretext tasks
- Pretext tasks for representation learning in NLP. An in-depth look into [BERT](https://arxiv.org/abs/1810.04805).


#### Exercise
In this exercise, you will train a small [BERT](https://arxiv.org/abs/1810.04805) model on the IMDB dataset (https://huggingface.co/datasets/imdb). You will then use the model to classify the sentiment of movie reviews and the sentiment of sentences from the Stanford Sentiment Treebank (SST2, https://huggingface.co/datasets/sst2).

## Week 4 - Contrastive Learning, SimCLR and mutual information-based proof 
- [Lecture](https://youtu.be/RlCqUawKcwA) || [Slides](https://uni-duesseldorf.sciebo.de/s/g5bD2N5QMpOM3lf) || [Notes](https://uni-duesseldorf.sciebo.de/s/zwbaz4mENfnuy28)
- A deep look into contrastive learning, theory,  and proof of MI bound.
- [SimCLR Paper](https://arxiv.org/abs/2002.05709)

#### Exercise
Build and train SimCLR resnet18 on CIFAR10.


## Week 5 - Understanding Contrastive learning & MoCO and image clustering
- [Lecture](https://youtu.be/PE1MT_S9m1k) || [Slides](https://uni-duesseldorf.sciebo.de/s/jZtCrfKRIRA2UmI) || [MoCO implementation](https://uni-duesseldorf.sciebo.de/s/NTnqx68EE630X4a)
- Contrastive Learning, L2 normalization, Properties of contrastive loss
- Momentum encoder (MoCO). Issues and concerns regarding batch normalization
- Multi-view contrastive learning
- Deep Image Clustering: task definition and challenges, K-means and [SCAN](https://arxiv.org/abs/2005.12320), [PMI and TEMI](https://arxiv.org/abs/2303.17896)

#### Exercise
Use pretrained MoCO ResNet50 for image clustering.


## Week 6 - Vision Transformers and Knowledge Distillation
- [Lecture](https://youtu.be/J_q-PEYikEo) || [Slides](https://uni-duesseldorf.sciebo.de/s/Jbx5bw87vlZrueB)
- Transformer encoder and Vision transformer
- ViTs VS CNNs: receptive field and inductive biases
- Knowledge distillation and the mysteries of model ensembles
- Knowledge distillation in ViTs and masked image modeling

#### Exercise
Knowledge distillation on CIFAR100 with Vision Transformers.

## Week 7 - Self-supervised learning without negative samples (BYOL, DINO)
- [Lecture](https://youtu.be/-VqXScgDZnM) || [Slides](https://uni-duesseldorf.sciebo.de/s/iU8owOBDx7PZdMs)
- A small review of self-supervised methods
- A small review of knowledge distillation
- Self-Supervised Learning & knowledge distillation
- An in-depth look into DINO

#### Exercise (2-week assignment)
In this exercise you will implement and train a DINO model on a medical dataset, the PathMNIST dataset from [medmnist](https://medmnist.com/) consisting of low-resolution images of various colon pathologies.


## Week 8 - Masked-based visual representation learning: MAE, BEiT, iBOT, DINOv2 
- [Lecture](https://youtu.be/8KP2SCm1YVo) || [Slides](https://uni-duesseldorf.sciebo.de/s/ifWDLlUdGRYBvD9)
- MAE: https://arxiv.org/abs/2111.06377
- BEiT: BERT-style pre-training in vision: https://arxiv.org/abs/2106.08254
- iBOT: Combining MIM with DINO https://arxiv.org/abs/2111.07832
- DINOv2: https://arxiv.org/abs/2304.07193


## Week 9 - Multimodal representation learning, robustness, and visual anomaly detection 
- [Lecture](https://youtu.be/eAf9UjPXmVg) || [Slides](https://uni-duesseldorf.sciebo.de/s/N8mAjFoMLaDJbHZ)
- Defining Robustness and Types of Robustness
- Zero-shot learning
- Contrastive Language Image Pretraining (CLIP)
- Image captioning
- Few-shot learning
- Visual anomaly detection: task definition
- Anomaly detection scores
- Anomaly detection metrics: AUROC


#### Exercise  (2-week assignment)
Use a CLIP-pre-trained model for out-of-distribution detection.


## Week 10 - Emerging properties of the learned representations and scaling laws
- [Lecture](https://youtu.be/CiXyNHVTxLs) || [Slides](https://uni-duesseldorf.sciebo.de/s/FMc1pDa9js5OTcR)
- Investigating CLIP models and scaling laws
- Determine factor of success of CLIP?
- How does CLIP scale to larger datasets and models?
- OpenCLIP: Scaling laws of CLIP models and connection to NLP scaling laws
- Robustness of CLIP models against image manipulations
- Learned representations of supervised models:CNNs VS Vision Transformers (ViTs), the texture-shape bias
- Robustness and generalization of supervised-pretrained CNNs VS ViTs
- Scaling (Supervised) Vision Transformers 
- [Properties of ViT pretrained models](https://theaisummer.com/vit-properties/)


## Week 11 - Investigating the self-supervised learned representations
- [Lecture](https://youtu.be/uUR0yEZ55Vg) || [Slides](https://uni-duesseldorf.sciebo.de/s/NALUEG5AlUzhbI3)
- Limitations of existing vision language models 
- Self-supervised VS supervised learned feature representations
- What do vision transformers (ViTs) learn “on their own”?
- MoCOv3 and DINO: https://arxiv.org/abs/2104.14294
- Self-supervised learning in medical imaging
- Investigating the pre-training self-supervised objectives

#### Exercise
No exercise takes place this week.

## Week 12 - Representation Learning in Proteins
- [Lecture](https://youtu.be/ZFazdK7dA7Q) || [Slides](https://uni-duesseldorf.sciebo.de/s/hFCXvnJCpAiPzlR)
- A closer look at the attention mechanism. The attention mechanism in Natural Language Translation
- A tiny intro to proteins
- Representing protein sequences with Transformers: BERT masked language modeling VS GPT?
- [ESM](https://www.pnas.org/doi/full/10.1073/pnas.2016239118), [ESMv2])(https://pubmed.ncbi.nlm.nih.gov/36927031/)
- [Looking & combining at the attention maps of a pre-trained Transformer](https://www.biorxiv.org/content/10.1101/2020.12.15.422761v1)
- [Protein Language models generalize beyond natural proteins](https://www.biorxiv.org/content/10.1101/2022.12.21.521521v1)

#### Exercise
Use a pretrained Protein Language Model


## Week 13 AlphaFold2
- [Lecture]() || [Slides]()

#### Exercise
[Just play around with an Alphafold notebook](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)


# Additional info
Feel free to open issues regarding errors on the exercises or missing information and we will try to get back to you. 
> Important: Solutions to the exercises are not provided, but you can cross-check your results with the Expected results in the notebook.
