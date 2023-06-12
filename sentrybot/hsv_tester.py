# import the necessary packages
import time

import cv2
import numpy as np


# initialize the camera and grab a reference to the raw camera capture
camera = cv2.VideoCapture(0)

while True:
    while True:
        try:
            hue_value = int(input("Hue value between 10 and 245: "))
            if (hue_value < 10) or (hue_value > 245):
                raise ValueError
        except ValueError:
            print("That isn't an integer between 10 and 245, try again")
        else:
            break

    lower_red = np.array([hue_value - 10, 100, 100])
    upper_red = np.array([hue_value + 10, 255, 255])

    while True:
        ret, image = camera.read()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color_mask = cv2.inRange(hsv, lower_red, upper_red)

        result = cv2.bitwise_and(image, image, mask=color_mask)

        cv2.imshow("Camera Output", image)
        cv2.imshow("HSV", hsv)
        cv2.imshow("Color Mask", color_mask)
        cv2.imshow("Final Result", result)

        k = cv2.waitKey(5)  # & 0xFF
        if "q" == chr(k & 255):
            break

    camera.release()
    cv2.destroyAllWindows()
