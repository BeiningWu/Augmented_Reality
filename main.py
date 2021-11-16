from Mode_SingleMark import mode_SingleMark
from Mode_RectMark import mode_RectMark
import cv2.cv2 as cv2
from time import sleep

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    while True:
        mode1 = mode_SingleMark(cap,'Markers')
        name = mode1.execute()
        sleep(2)
        print('123')
        cap.release()
        cv2.destroyWindow(name)
        break
