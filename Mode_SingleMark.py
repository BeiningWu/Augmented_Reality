import cv2.cv2 as cv2
import cv2.aruco as aruco
import numpy as np
import os


class mode_SingleMark:
    def __init__(self, cap, path):
        self.cap = cap
        self.path = path

    def __del__(self):
        print('__del__')

    def augmentAruco(self, bbox, id, img, imgAug, drawId=True):
        tl = bbox[0][0][0], bbox[0][0][1]
        tr = bbox[0][1][0], bbox[0][1][1]
        br = bbox[0][2][0], bbox[0][2][1]
        bl = bbox[0][3][0], bbox[0][3][1]

        h, w, c = imgAug.shape
        pts1 = np.array([tl, tr, br, bl])
        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matirx, _ = cv2.findHomography(pts2, pts1)
        imgOut = cv2.warpPerspective(imgAug, matirx, (img.shape[1], img.shape[0]))
        cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
        imgOut = img + imgOut
        if drawId:
            cv2.putText(imgOut, str(id), (np.uint32(tl[0]).item(), np.uint32(tl[1]).item()), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)
        return imgOut

    def findArucoMarkers(self, img, markerSize=6, totalMarkers=250, draw=True):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
        arucoDict = aruco.Dictionary_get(key)
        arucoParam = aruco.DetectorParameters_create()
        bboxs, ids, rejected= aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
        print(rejected)

        # print(ids)
        if draw:
            aruco.drawDetectedMarkers(img, bboxs)

        return bboxs, ids

    def loadAugImages(self):
        myList = os.listdir(self.path)
        augDicts = {}
        for imgPath in myList:
            key = int(os.path.splitext(imgPath)[0])
            imgAug = cv2.imread(f'{self.path}/{imgPath}')
            augDicts[key] = imgAug
        return augDicts

    def execute(self):
        augDicts = self.loadAugImages()

        while True:
            success, img = self.cap.read()
            arucoFound0, arucoFound1 = self.findArucoMarkers(img)

            if len(arucoFound0) != 0:
                for bbox, id in zip(arucoFound0, arucoFound1):
                    if int(id) in augDicts:
                        img = self.augmentAruco(bbox, id, img, augDicts[int(id)])

            self.findArucoMarkers(img)
            cv2.imshow('Image', img)
            k = cv2.waitKey(1)
            if k == 27:  # Esc# key to stop
                break
        return 'Image'
