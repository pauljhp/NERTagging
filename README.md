# NERTagging
NER Tagging with transformer - for HKU COMP7607 NLP course

## Setup

- Run `$ conda env create -n nertag_env --file environment.yml` to install the dependencies.

- Download datasets through `$ bash download_data.sh`

## Training

- Modify parameters in `train_transformer.sh` and `train_lstm.sh`, then run the scripts.

- Training logs will be written into `./log/traininglog` and models saved at `./checkpoints/`