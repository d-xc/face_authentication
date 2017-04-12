#!/usr/bin/env bash

./align-dlib.py images/ align outerEyesAndNose aligned-images/ --size 96

rm aligned-images/cache.t7

./main.lua -outDir embeddings/ -data aligned-images/

./classifier.py train embeddings/

cp embeddings/classifier.pkl ..
