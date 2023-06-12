import cv2
import numpy as np

MINIMUM_RANGE = 0
MAXIMUM_RANGE = 255


def main(minimum_hue, maximum_hue):
    video_capture = cv2.VideoCapture(0)  # Change to PiCamera for SentryBot

    current_max_area = 0
    current_center_x = 0
    current_center_y = 0

    while True:
        _, frame = video_capture.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([minimum_hue, MINIMUM_RANGE, MAXIMUM_RANGE])
        upper_bound = np.array([maximum_hue, MAXIMUM_RANGE, MAXIMUM_RANGE])

        colour_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(
            colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            position_x, position_y, width, height = cv2.boundingRect(contour)
            area = width * height
            center_x = position_x + width / 2
            center_y = position_y + height / 2

            if area > current_max_area:
                current_max_area = area
                current_center_x = center_x
                current_center_y = center_y

        if current_max_area > 0:
            print(f"{current_max_area=}")
            print(f"{current_center_x=}")
            print(f"{current_center_y=}")


if __name__ == "__main__":
    main(30, 50)

# Between 50K and 70K
