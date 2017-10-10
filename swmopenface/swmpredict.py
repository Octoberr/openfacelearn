# coding:utf-8
import os
import openface
import cv2
import pickle
import numpy as np

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
classifierDir = os.path.join(fileDir, '..', 'generated-embeddings')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predict = os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat')
torchmodel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
align = openface.AlignDlib(predict)
net = openface.TorchNeuralNet(torchmodel)
landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
predictdatabase = os.path.join(classifierDir, 'classifier.pkl')  # 人脸数据库


# 获取人脸的处理
def getRep(img):
    bgrImg = cv2.imread(img)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bbs = align.getAllFaceBoundingBoxes(rgbImg)
    reps = []
    for bb in bbs:
        facelandmarks = align.findLandmarks(rgbImg, bb)
        alignedFace = align.align(96, rgbImg, bb, facelandmarks, landmarkIndices=landmarkIndices)
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


if __name__ == '__main__':
    with open(predictdatabase, 'rb') as f:
        (le, clf) = pickle.load(f)
    # img = '2.png'  # 一个人
    img = 'swmandzyh1.jpg'  # 多个人
    reps = getRep(img)
    data = []
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        print("It is {}.".format(person.decode('utf-8')))
        data.append((person.decode('utf-8'), confidence))
    print(data)




