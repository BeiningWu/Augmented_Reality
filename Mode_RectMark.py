import cv2.cv2 as cv2
import cv2.aruco as aruco
import numpy as np
import os

from Mode_SingleMark import mode_SingleMark


class mode_RectMark(mode_SingleMark):
    def __init__(self, cap, path):
        mode_SingleMark.__init__(self, cap, path)

    # def locatePoint(self, arucoFound):
    #
    #
    #
    #     return result
    def augmentAruco(self, img, imgAug, tl, tr, bl, br):

        h, w, c = imgAug.shape
        pts1 = np.array([tl, tr, br, bl])
        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matirx, _ = cv2.findHomography(pts2, pts1)
        imgOut = cv2.warpPerspective(imgAug, matirx, (img.shape[1], img.shape[0]))
        cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
        imgOut = img + imgOut
        return imgOut

    def execute(self):
        while True:
            success, img = self.cap.read()
            augImg = cv2.imread('p1.jpg')
            arucoFound0, arucoFound1 = self.findArucoMarkers(img)

            if len(arucoFound0) != 0 and len(arucoFound1) >= 4:
                for bbox, id in zip(arucoFound0, arucoFound1):
                    if int(id) == 0:
                        tl = bbox[0][0][0], bbox[0][0][1]
                    elif int(id) == 1:
                        tr = bbox[0][0][0], bbox[0][0][1]
                    elif int(id) == 2:
                        br = bbox[0][0][0], bbox[0][0][1]
                    elif int(id) == 3:
                        bl = bbox[0][0][0], bbox[0][0][1]

                img = self.augmentAruco(img, augImg, tl, tr, bl, br)

            cv2.imshow('Image', img)

            k = cv2.waitKey(1)
            if k == 27:  # Esc# key to stop
                break
        return 'Image'

# from Mode_SingleMark import mode_SingleMark
#
#
# class mode_RectMark(mode_SingleMark):
#     def __init__(self, cap, path):
#         mode_SingleMark.__init__(self, cap, path)
#
#     # def locatePoint(self, arucoFound):
#     #
#     #
#     #
#     #     return result
#     def augmentAruco(self, img, imgAug, tl, tr, bl, br):
#
#         h, w, c = imgAug.shape
#         pts1 = np.array([tl, tr, br, bl])
#         pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
#         matirx, _ = cv2.findHomography(pts2, pts1)
#         imgOut = cv2.warpPerspective(imgAug, matirx, (img.shape[1], img.shape[0]))
#         cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
#         imgOut = img + imgOut
#         return imgOut
#
#     def execute(self):
#         while True:
#             success, img = self.cap.read()
#             augImg = cv2.imread('p1.jpg')
#             arucoFound0, arucoFound1 = self.findArucoMarkers(img)
#
#             if len(arucoFound0) != 0 and len(arucoFound1) >= 4:
#                 arucoFound = np.array(arucoFound0[:4]).reshape(16, 2)
#
#                 xnum = np.average(arucoFound[:, 0])
#                 ynum = np.average(arucoFound[:, 1])
#                 pointList = (arucoFound[:, 0] - xnum) ** 2 + (arucoFound[:, 1] - ynum) ** 2
#                 pointIndexList = pointList.argsort()[-4:]
#                 pointList = arucoFound[pointIndexList]
#
#                 distanceList = pointList[:, 0] ** 2 + pointList[:, 1] ** 2
#                 tl = pointList[distanceList.argsort()[0]]
#                 br = pointList[distanceList.argsort()[-1]]
#
#                 xLength = img.shape[0]
#                 distanceList = (pointList[:, 0] - xLength) ** 2 + pointList[:, 1] ** 2
#                 tr = pointList[distanceList.argsort()[0]]
#                 bl = pointList[distanceList.argsort()[-1]]
#
#                 img = self.augmentAruco(img, augImg, tl, tr, bl, br)
#
#             cv2.imshow('Image', img)
#
#             k = cv2.waitKey(1)
#             if k == 27:  # Esc# key to stop
#                 break
#         return 'Image'
