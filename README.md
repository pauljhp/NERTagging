# NERTagging
NER Tagging with transformer - for HKU COMP7607 NLP course

## Setup

- Run `$ conda env create -n nertag_env --file environment.yml` to install the dependencies.

- Download datasets through `$ bash download_data.sh`

## Training

- Modify parameters in `train_transformer.sh` and `train_lstm.sh`, then run the scripts.

- Training logs will be written into `./log/traininglog` and models saved at `./checkpoints/`

## Inference

- Run `$ bash inference.sh` for producing the test files

## Others

- This repository is for my coursework at HKU COMP7607 NLP course. I've also made it available on [GitHub](https://github.com/pauljhp/NERTagging)