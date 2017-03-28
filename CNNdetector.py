#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import cv2
import numpy as np
from time import time
import dlib

def bbreg(boundingbox, reg):
    reg = reg.T

    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print "reshape of reg"
        pass

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1

    bb0 = boundingbox[:, 0] + reg[:, 0] * w
    bb1 = boundingbox[:, 1] + reg[:, 1] * h
    bb2 = boundingbox[:, 2] + reg[:, 2] * w
    bb3 = boundingbox[:, 3] + reg[:, 3] * h

    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    return boundingbox

def pad(boxesA, w, h):
    boxes = boxesA.copy()  # shit, value parameter!!!

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy - 1).astype(np.int32)
    dx = np.maximum(0, dx - 1).astype(np.int32)
    y = np.maximum(0, y - 1).astype(np.int32)
    x = np.maximum(0, x - 1).astype(np.int32)
    edy = np.maximum(0, edy - 1).astype(np.int32)
    edx = np.maximum(0, edx - 1).astype(np.int32)
    ey = np.maximum(0, ey - 1).astype(np.int32)
    ex = np.maximum(0, ex - 1).astype(np.int32)

    return [
        dy, edy, dx, edx, y, ey, x, ex,
        tmpw.astype(np.int32), tmph.astype(np.int32)]

def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T
    return bboxA

def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())  # read s using I

    pick = []
    while len(I):
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    return pick

def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0, :, :].T
    dy1 = reg[1, :, :].T
    dx2 = reg[2, :, :].T
    dy2 = reg[3, :, :].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x

    score = map[x, y]
    reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    # matlab index from 1, so with "boundingbox-1"
    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T
    # while python don't have to
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    return boundingbox_out.T


def stage1(img, net, threshold, minsize, scaleFactor):
    total_boxes = np.zeros((0, 9), np.float32)
    h = img.shape[0]
    w = img.shape[1]
    minl = min(img.shape[0], img.shape[1])
    m = 12.0 / minsize
    minl = minl * m

    # create scale pyramid
    scales = []
    factor_count = 0
    while minl >= 12:
        scales.append(m * pow(scaleFactor, factor_count))
        minl *= scaleFactor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))

        im_data = cv2.resize(img, (ws, hs))
        im_data = im_data.astype(np.float32)
        im_data = (im_data - 127.5) * 0.0078125

        net.blobs['data'].reshape(1, 3, ws, hs)
        net.blobs['data'].data[...] = np.expand_dims(
            im_data.transpose(2, 1, 0), axis=0)
        out = net.forward()

        boxes = generateBoundingBox(
            out['prob1'][0, 1, :, :], out['conv4-2'][0], scale, threshold)
        if boxes.shape[0] != 0:
            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0:
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
    return total_boxes

def detect_face(img, minsize, PNet, RNet, ONet, threshold, factor):
    #####
    # 1 #
    #####
    h = img.shape[0]
    w = img.shape[1]

    total_boxes = stage1(img, PNet, threshold[0], minsize, factor)
    # print "[1]:", total_boxes.shape[0]
    points = np.zeros(0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        # print "[2]:", total_boxes.shape[0]

        # revise and convert to square
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        t1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        t2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        t3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        t4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        t5 = total_boxes[:, 4]
        total_boxes = np.array([t1, t2, t3, t4, t5]).T

        total_boxes = rerec(total_boxes)  # convert box to square
        # print "[4]:", total_boxes.shape[0]

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
        # print "[4.5]:", total_boxes.shape[0]
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3))
        for k in range(numbox):
            tmp = np.zeros((tmph[k], tmpw[k], 3))
            tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1] = img[y[k]:ey[k]+1, x[k]:ex[k]+1]
            tempimg[k, :, :, :] = cv2.resize(tmp, (24, 24))

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg = np.swapaxes(tempimg, 1, 3)

        # RNet
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        score = out['prob1'][:, 1]
        pass_t = np.where(score > threshold[1])[0]

        score = np.array([score[pass_t]]).T
        total_boxes = np.concatenate(
            (total_boxes[pass_t, 0:4], score), axis=1)
        # print "[5]:", total_boxes.shape[0]

        mv = out['conv5-2'][pass_t, :].T
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            if len(pick) > 0:
                total_boxes = total_boxes[pick, :]
                # print "[6]:",total_boxes.shape[0]
                total_boxes = bbreg(total_boxes, mv[:, pick])
                # print "[7]:",total_boxes.shape[0]
                total_boxes = rerec(total_boxes)
                # print "[8]:",total_boxes.shape[0]

        #####
        # 2 #
        #####
        # print "2:", total_boxes.shape

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage

        total_boxes = np.fix(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

        tempimg = np.zeros((numbox, 48, 48, 3))
        for k in range(numbox):
            tmp = np.zeros((tmph[k], tmpw[k], 3))
            tmp[dy[k]:edy[k] + 1, dx[k]:edx[k] + 1] = img[y[k]:ey[k] + 1, x[k]:ex[k] + 1]
            tempimg[k, :, :, :] = cv2.resize(tmp, (48, 48))
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg = np.swapaxes(tempimg, 1, 3)

        # ONet
        ONet.blobs['data'].reshape(numbox, 3, 48, 48)
        ONet.reshape()
        ONet.blobs['data'].data[...] = tempimg
        out = ONet.forward()

        score = out['prob1'][:, 1]
        points = out['conv6-3']
        pass_t = np.where(score > threshold[2])[0]
        points = points[pass_t, :]
        score = np.array([score[pass_t]]).T
        total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
        # print "[9]:", total_boxes.shape[0]

        mv = out['conv6-2'][pass_t, :].T
        w = total_boxes[:, 3] - total_boxes[:, 1] + 1
        h = total_boxes[:, 2] - total_boxes[:, 0] + 1

        points[:, 0:5] = np.tile(w, (5, 1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5, 1)).T - 1
        points[:, 5:10] = np.tile(h, (5, 1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T - 1

        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes, mv[:, :])
            # print "[10]:", total_boxes.shape[0]
            pick = nms(total_boxes, 0.7, 'Min')

            if len(pick) > 0:
                total_boxes = total_boxes[pick, :]
                # print "[11]:", total_boxes.shape[0]
                points = points[pick, :]
    #####
    # 3 #
    #####
    # print "3:", total_boxes.shape

    return total_boxes, points


def testONet(ONet):
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        t1 = time()
        for j in range(40):
            ONet.blobs['data'].reshape(i + j % 2, 3, 48, 48)
            ONet.reshape()
            ONet.blobs['data'].data[...] = 0
            ONet.forward()
        t2 = time()
        print i, (t2 - t1) / 40.0, (i + 0.5) * 40 / (t2 - t1)


def a():
    pass


class mtcnnFaceDetector(object):
    def __init__(self, model_path, minSize=80):
        caffe.set_mode_gpu()
        self.minSize = minSize
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.PNet = caffe.Net(
            model_path + '/det1.prototxt',
            model_path + '/det1.caffemodel', caffe.TEST)
        self.RNet = caffe.Net(
            model_path + '/det2.prototxt',
            model_path + '/det2.caffemodel', caffe.TEST)
        self.ONet = caffe.Net(
            model_path + '/det3.prototxt',
            model_path + '/det3.caffemodel', caffe.TEST)
        #testONet(self.ONet)

    def __call__(self, img):
        img_matlab = img[:, :, ::-1].copy()
        boundingboxes, points = detect_face(
            img_matlab, self.minSize,
            self.PNet, self.RNet, self.ONet, self.threshold, self.factor)

        boundingboxes = boundingboxes.astype(int)

        # The second paprameter is upscale factor
        dets = []
        for k, bbox in enumerate(boundingboxes):
            dets.append(
                dlib.rectangle(
                    left=bbox[0],
                    top=bbox[1],
                    right=bbox[2],
                    bottom=bbox[3]))
        return dets, points
