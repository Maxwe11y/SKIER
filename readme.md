SKIER: A Symbolic Knowledge Integrated Model for Conversational Emotion Recognition
=========

## Overview
>In this repo, we put the data and code of the SKIER model for emotion recognition in conversations(ERC). 
>We propose a symbolic knowledge integrated model for the ERC task, named SKIER, which effectively leverages symbolic-based dependency knowledge at the utterance level, and commonsense knowledge at the concept level.
> We introduce a dialogue relation graph-based contextualizer for SKIER to functionally fuse utterance dependencies. Meanwhile, we propose
a relation-aware concept representation mechanism to integrate the concepts in different relations.
> Our method achieves state-of-the-art performance on the ERC task.



<img src="https://github.com/Maxwe11y/SKIER/blob/main/model_cr.png" width = 90% height = 90% div align=center />

<!-- <figure class="half">
  <img src="https://github.com/Maxwe11y/SKIER/blob/main/model_cr.png" width = 90% height = 90% div align=left />
  <img src="https://github.com/Maxwe11y/SKIER/blob/main/model_cr.png" width = 90% height = 90% div align=right />
</figure> -->

### Data DIR Structure

The structure of the data dir is as follows:

```
data
└───   EMORY
       └─── EMORY.pkl
       └─── EMORY_revised.pkl
└───   dailydialog
       └─── Daily.zip
       └─── Daily_revised.zip
       └─── daily_.json
└───   dialog_concept
       └─── causes_weight_dict_all.json
       └─── hascontext_weight_dict_all.json
       └─── isa_dict_all.json
       └─── isa_weight_dict_all.json
```
 MELD(Plz check out the following sharing link)
 * [MELD_revised](https://www.dropbox.com/s/edspgpbgnouh21h/MELD_revised.zip?dl=0)
 * [MELD](https://www.dropbox.com/s/5m6rcg5g2nhys22/MELD.zip?dl=0)



## Uasge
In order to implement the proposed SKIER framework, you have to download the pre-trained GloVe vectors(glove.6B.100d.txt is the most commonly used vectors in this project).
The downloaded GloVe vectors should be placed in glove dir(plz create glove dir if empty). Note that the batch size should be set to 1 as we process one dialogue each time.

👉 Check out [GloVe Embeddings](https://nlp.stanford.edu/data/glove.6B.zip) before you run the **code**.


To run this code, plz use the following command (take dailydialogue dataset as an example)
```
python3 train_dd.py --model-type roberta_large --att_dropout 0.5 --output_dim 1024 --chunk_size 50 --base-lr 0.0000005  --epochs 15 --num_epochs 40 --num_relations 11 --data_type daily --num_features 3 --freeze_glove --num_class 7 --use_fixed
```


## Citation
Please cite as
```bibtex

@article{Li_Zhu_Mao_Cambria_2023,
title={{SKIER}: A Symbolic Knowledge Integrated Model for Conversational Emotion Recognition}, 
volume={37}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/26541}, 
DOI={10.1609/aaai.v37i11.26541}, 
number={11}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Li, Wei and Zhu, Luyao and Mao, Rui and Cambria, Erik}, 
year={2023}, 
month={Jun.}, 
pages={13121-13129}
}

```

