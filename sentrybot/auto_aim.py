import cv2
import numpy as np

MINIMUM_RANGE = 0
MAXIMUM_RANGE = 255

# MINIMUM_TARGET_AREA = 50000
# MAXIMUM_TARGET_AREA = 60000

MINIMUM_TARGET_AREA = 0
MAXIMUM_TARGET_AREA = 100000


def draw_contour(image, current_contour):
    cv2.drawContours(image, [current_contour], 0, (0, 255, 0), 3)


def contour_to_rectangle(contour):
    polygonal_curve = cv2.approxPolyDP(contour, 3, True)
    bounding_rectangle = cv2.boundingRect(polygonal_curve)

    return (
        bounding_rectangle[0],
        bounding_rectangle[1],
        bounding_rectangle[2],
        bounding_rectangle[3],
    )


def draw_rectangle(
    image, current_position_x, current_position_y, current_width, current_height
):
    cv2.rectangle(
        image,
        pt1=(current_position_x, current_position_y),
        pt2=(current_position_x + current_width, current_position_y + current_height),
        color=(255, 0, 0),
        thickness=3,
    )


def draw_contour_box(image, current_contour):
    (
        contour_x,
        contour_y,
        contour_width,
        contour_height,
    ) = contour_to_rectangle(current_contour)
    draw_rectangle(
        image,
        contour_x,
        contour_y,
        contour_width,
        contour_height,
    )
    draw_contour(image, current_contour)

    print(f"{contour_x=}")
    print(f"{contour_y=}")
    print(f"{contour_width=}")
    print(f"{contour_height=}")


def diagnostic_plots(frame_list, current_contour):
    for index, image in enumerate(frame_list):
        draw_contour_box(image, current_contour)

        cv2.imshow(f"image_{index}", image)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == ord("x"):
        return


def main(minimum_hue, maximum_hue):
    # Change to PiCamera for SentryBot
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", width=600, height=200)

    image_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"{image_width=}")
    print(f"{image_height=}")

    image_center_x = image_width / 2
    image_center_y = image_height / 2

    current_max_area = 0
    current_position_x = 0
    current_position_y = 0
    current_center_x = 0
    current_center_y = 0
    current_width = 0
    current_height = 0
    current_contour = None

    while True:
        _, frame = video_capture.read()

        # TODO: Check effectiveness of this.
        # frame = cv2.blur(frame, (3, 3))

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([minimum_hue, MINIMUM_RANGE, MAXIMUM_RANGE])
        upper_bound = np.array([maximum_hue, MAXIMUM_RANGE, MAXIMUM_RANGE])

        colour_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(
            colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            position_x, position_y, width, height = contour_to_rectangle(contour)
            area = width * height
            center_x = position_x + width / 2
            center_y = position_y + height / 2

            if area > current_max_area:
                current_max_area = area
                current_center_x = center_x
                current_center_y = center_y
                current_width = width
                current_height = height

                current_position_x = position_x
                current_position_y = position_y

                current_contour = contour

        if (
            current_max_area > MINIMUM_TARGET_AREA
            and current_max_area < MAXIMUM_TARGET_AREA
        ):
            print(f"{current_center_x=}")
            print(f"{current_center_y=}")
            print(f"{current_max_area=}")
            print(f"{image_width=}")

            draw_contour(frame, current_contour)

            if current_center_x > (image_center_x + image_width / 3):
                print("Object right")
            elif current_center_x < (image_center_x - image_width / 3):
                print("Object left")
            else:
                print("Object at the center")

        cv2.imshow("Auto-aiming", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main(30, 50)

# Between 50K and 60K
