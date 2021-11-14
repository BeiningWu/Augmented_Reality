from Mode_SingleMark import mode_SingleMark
import cv2.cv2 as cv2

if __name__ == "__main__":
    cap=cv2.VideoCapture(2)
    mode1 = mode_SingleMark(cap, 'Markers')
    mode1.execute()
