# Enabling Patient-side Disease Prediction via the Integration of Patient Narratives

This is the repository for WWW 2024 short paper **Enabling Patient-side Disease Prediction via the Integration of Patient Narratives**. 

In this paper, we propose Personalized Medical Disease Prediction (PoMP), which predicts diseases using patient health narratives including textual descriptions and demographic information. 
By applying PoMP, patients can gain a clearer comprehension of their conditions, empowering them to directly seek appropriate medical specialists and thereby reducing the time spent navigating healthcare communication to locate suitable doctors.

## Requirements

- colorama==0.4.4
- numpy==1.21.5
- pandas==1.3.5
- scikit_learn==0.23.1
- torch==1.13.0+cu116
- tqdm==4.64.0
- transformers==4.18.0

## Data preparation

In light of privacy considerations, our dataset's public release is currently limited to 5 examples per disease category. We plan to make the complete dataset available at a later date.

## Quick Start

```shell
python main.py --attention_dim 64 \
--batch_size 4 \
--data_path 'data/' \
--device 'cuda:0' \
--dropout 0.1 \
--epochs 30 --eval_metric 1 \
--gender_dim 32 \
--hierarchical_loss_weight 0.7 \
--in_dim 3 \
--n_heads 4 \
--pregnancy_dim 32 \
--learning_rate 0.00005 \
--model 'sentence-transformers/all-MiniLM-L6-v2' \
--tokenizer 'sentence-transformers/all-MiniLM-L6-v2'
```
