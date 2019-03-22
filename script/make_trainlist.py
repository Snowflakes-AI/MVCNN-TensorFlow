import os
from datetime import datetime
import random

SRCDIR='/path/to/save/folder'
CLASSLIST = 'classes.txt'
RATIO_HOLDOUT = 1./10
phase = ('train', 'test')

TRAIN='train.txt'
VAL='val.txt'
TEST='test.txt'

with open(CLASSLIST, 'r') as fs:
    classes = fs.read().split()

try:
    fsTrain = open(TRAIN, 'w')
    fsVal = open(VAL, 'w')
    fsTest = open(TEST, 'w')
except:
    print("File IO error")

for cls_id, cls in enumerate(classes):
    random.seed(datetime.now())
    for p in phase:
        path = os.path.join(SRCDIR, p, cls)
        imglist = os.listdir(path)
        if p == 'train':
            trainList = []
            valList = []
            min_vals = int(len(imglist) * RATIO_HOLDOUT)
            for img in imglist:
                seed = random.random()
                if seed > RATIO_HOLDOUT:
                    trainList.append(img)
                else:
                    valList.append(img)

            remain = min_vals - len(valList)
            if remain > 0:
                for i in range(remain):
                    valList.append(trainList.pop())
            print(cls_id, 'train %d / val %d' % (len(trainList), len(valList)))
            for img in trainList:
                fsTrain.write(os.path.join(path, img) + ' ' + str(cls_id) + '\n')
            for img in valList:
                fsVal.write(os.path.join(path, img) + ' ' + str(cls_id) + '\n')
        else:
            print(cls_id, 'test %d' % len(imglist))
            for img in imglist:
                fsTest.write(os.path.join(path, img) + ' ' + str(cls_id) + '\n')

fsTrain.close()
fsVal.close()
fsTest.close()
