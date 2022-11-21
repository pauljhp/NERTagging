#!/bin/bash

# conll data (already in repository)
curl https://data.deepai.org/conll2003.zip > ./data/conll2003.zip
unzip ./data/conll2003.zip -d ./data/conll2003/

# dowonload glove data
curl https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip > ./embeddings/glove.6B.zip
unzip ./embeddings/glove.6B.zip -d ./embeddings/glove