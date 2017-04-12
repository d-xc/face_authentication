#!/usr/bin/env python

import os
import cv2
import openface
import argparse
import time
import pickle
import numpy as np
import rospy
import thread
from std_msgs.msg import Int8
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from sklearn.mixture import GMM

from cv_bridge import CvBridge, CvBridgeError

fileDir = os.path.dirname(os.path.realpath(__file__))
dataDir = os.path.join(fileDir, '..', 'data') 
modelDir = os.path.join(dataDir, 'models')

needpending = 0
bridge = CvBridge()


parser = argparse.ArgumentParser()
    
parser.add_argument('--dlibFacePredictor', 
                type=str, 
                help="Path to dlib's face predictor.",
                default=os.path.join(modelDir, 'dlib', 'shape_predictor_68_face_landmarks.dat'))
parser.add_argument('--networkModel',
                 type=str,
                 help="Path to Torch network model.",
                 default=os.path.join(modelDir, 'openface', 'nn4.small2.v1.t7'))
parser.add_argument('--classifierModel',
                     type=str,
                     help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.',
                     default=os.path.join(dataDir, 'classifier.pkl'))
parser.add_argument('--imgDim', type=int,
                help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--multi', help="Infer multiple faces in image", action="store_true", default=False)
parser.add_argument('--verbose', action='store_true')

#args = parser.parse_args() 
args = parser.parse_args(rospy.myargv()[1:])

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)

def getRep(rgbImg, multiple=False):
    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        print("Unable to find a face")
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print("Unable to align image")
        if args.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        if args.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def train(args):
    pass

def infer(args, multiple, rgbImg, le, clf):
    reps = getRep(rgbImg, multiple)
    if len(reps) > 1:
        print("List of faces in image from left to right")
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
        if multiple:
            print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                         confidence))
        else:
            print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("+ Distance from the mean: {}".format(dist))             
    return person, confidence

def recog_thread(frame):
    global needpending
    person, confidence = infer(args, args.multi, frame, le, clf) 
    if confidence > 0.85:
        autherizedFlag= 1
        pub.publish(autherizedFlag)
        needpending = 1
        cv2.destroyWindow('frame')
        return
    else:
        autherizedFlag = 0
        pub.publish(autherizedFlag)

frame = 0
def cb_image(data):
    global needpending
    global frame

    if needpending == 1:
        return;
    bgrImg = bridge.imgmsg_to_cv2(data, "bgr8")
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    frameSmall = cv2.resize(rgbImg, (160, 120))
    bb = align.getLargestFaceBoundingBox(frameSmall)
    if bb == None:
        if args.verbose:
            print("No face is found in this frame.")
    else:
        if args.verbose:
            print("Face locates at(top={}, bottom={}, left={}, right={})").format(bb.top(),bb.bottom(),bb.left(),bb.right())
        bl = (int(bb.left()*4), int(bb.bottom()*4))
        tr = (int(bb.right()*4), int(bb.top()*4))
        cv2.rectangle(bgrImg, bl, tr, color=(153, 255, 204),thickness=3)
        if  frame % 10 == 0:
           thread.start_new_thread(recog_thread, (frameSmall, ))
        frame += 1
    if needpending != 1:    
        cv2.imshow('frame', bgrImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

def cb_cmd(data):
    global needpending
    if data.data == 5:
        needpending = 0

if __name__ == '__main__':
    
    rospy.init_node('face_auth', anonymous=True)
    pub = rospy.Publisher('face/auth', Int8, queue_size=5)
#    rate = rospy.Rate(10)
    
      
    
    start = time.time()

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()

#    cap = cv2.VideoCapture(0)
    
    #open the classifierModel
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f) 
      
    #authentication flag
    autherizedFlag = 0  
    
    rospy.Subscriber("/camera/color/image_raw", Image, cb_image)
    rospy.Subscriber("/voice/cmd_topic", Int32, cb_cmd)
    rospy.spin()
 #   while not rospy.is_shutdown():
 #       ret, bgrImg = cap.read()
 #       rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
 #       bb = align.getLargestFaceBoundingBox(rgbImg)
 #       if bb == None:
 #           if args.verbose:
 #               print("No face is found in this frame.")
 #       else:
 #           if args.verbose:
  #              print("Face locates at(top={}, bottom={}, left={}, right={})").format(bb.top(),bb.bottom(),bb.left(),bb.right())
 #           bl = (bb.left(),bb.bottom())
 #           tr = (bb.right(), bb.top())
 #           cv2.rectangle(bgrImg, bl, tr, color=(153, 255, 204),thickness=3)
 #           person, confidence = infer(args, args.multi, rgbImg, le, clf)
 #           if confidence > 0.8:
 #               autherizedFlag= 1
 #           else:
 #               autherizedFlag = 0
 #           pub.publish(autherizedFlag)
              
 #       cv2.imshow('frame', bgrImg)
        
 #       rate.sleep()
        
 #       if cv2.waitKey(1) & 0xFF == ord('q'):
 #           break
        
    # When everything done, release the capture
 #   cap.release()
 #   cv2.destroyAllWindows()

    


