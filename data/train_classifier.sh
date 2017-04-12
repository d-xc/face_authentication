#!/usr/bin/env bash


~/projects/openface/util/align-dlib.py ~/projects/openface/images/training-images/ align outerEyesAndNose ~/projects/openface/images/aligned-training-images/ --size 96

rm /home/joanne/projects/openface/images/aligned-training-images/cache.t7

~/projects/openface/batch-represent/main.lua -outDir ~/projects/openface/images/generated-embeddings/ -data ~/projects/openface/images/aligned-training-images/

~/projects/openface/demos/classifier.py train ~/projects/openface/images/generated-embeddings/

cp ~/projects/openface/images/generated-embeddings/classifier.pkl ~/eclipse_workspace/FaceRecognitionDemo/src/face_authentication/data
