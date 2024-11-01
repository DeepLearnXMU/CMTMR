# Cross-Modal Text-Molecule Retrieval

This repository contains the code for our paper “Towards Cross-Modal Text-Molecule Retrieval with Better Modality Alignment” (BIBM 2024 regular paper).

Our implementation is built on the source code from [text2mol](https://github.com/cnedwards/text2mol), [MoMu-GraphTextRetrieval](https://github.com/yangzhao1230/GraphTextRetrieval/tree/631e6b7e31a7e2ef96c3724a9b663c70cc9bb7e2) and [ACME](https://github.com/hwang1996/ACME). Thanks for their work. 

## Dataset

We use ChEBI-20 dataset from [text2mol](https://github.com/cnedwards/text2mol) to conduct the main experiment and PCdes dataset from [KV-PLM](https://github.com/thunlp/KV-PLM/tree/master) to conduct comparison with pretrain-finetune paradigm based models.

You need to download the ChEBI-20 dataset from [text2mol](https://github.com/cnedwards/text2mol) and put it in the `data_dir`.

## How to Run?

To train and test our model, you can simply run:

```bash
bash scripts/train.sh
```

The model is tested after 60 epochs have been trained, so you can get the results of the text-to-molecule retrieval.

To finetune a trained model on kv_data with paragraph-level and testing:

```bash
bash scripts/finetune_para.sh
```

To finetune a trained model on kv_data with sentence-level and testing:

```bash
bash scripts/finetune_sent.sh
```

