import numpy as np
import cv2

cap = cv2.VideoCapture(0)


def nothing(x):
    pass


cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", (600, 200))
cv2.createTrackbar("minH", "image", 0, 255, nothing)
cv2.createTrackbar("minS", "image", 0, 255, nothing)
cv2.createTrackbar("minV", "image", 0, 255, nothing)
cv2.createTrackbar("maxH", "image", 0, 255, nothing)
cv2.createTrackbar("maxS", "image", 0, 255, nothing)
cv2.createTrackbar("maxV", "image", 0, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV
    )  # convert to hsv encoding for better processing

    minH = cv2.getTrackbarPos("minH", "image")
    minS = cv2.getTrackbarPos("minS", "image")
    minV = cv2.getTrackbarPos("minV", "image")
    maxH = cv2.getTrackbarPos("maxH", "image")
    maxS = cv2.getTrackbarPos("maxS", "image")
    maxV = cv2.getTrackbarPos("maxV", "image")

    a = cv2.waitKey(5) & 0xFF
    if a == ord("p"):
        print(
            "minH: ",
            minH,
            "\nmaxH: ",
            maxH,
            "\nminS : ",
            minS,
            "\nmaxS : ",
            maxS,
            "\nminV : ",
            minV,
            "\nmaxV : ",
            maxV,
        )
    lowerpink = np.array([minH, minS, minV])
    upperpink = np.array([maxH, maxS, maxV])
    # print(lowerpink + '\n' + upperpink)

    mask = cv2.inRange(hsv, lowerpink, upperpink)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # median = cv2.bilateralFilter(res,15,75,75)

    # cv2.imshow('median',descale(median,3))
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
