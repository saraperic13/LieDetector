import cv2
import numpy


class BlushingDetector:
    # indexes of the facial landmarks for the left and right cheek
    right_cheek_idx = [1, 2, 3, 4, 48, 31, 36]
    left_cheek_idx = [12, 13, 14, 15, 45, 35, 54]

    RED_CHANGE_ALLOWANCE = 35
    GREEN_CHANGE_ALLOWANCE = 100
    BLUE_CHANGE_ALLOWANCE = 80
    RGB_CHANGE_ALLOWANCE = 60

    def __init__(self, blushing_consecutive_frames=50):
        # counter of frames in which blushing occurred
        self.blushing_frame_counter = 0
        # the number of consecutive frames the blushing should occur to be detected
        self.BLUSHING_CONSECUTIVE_FRAMES = blushing_consecutive_frames
        # calculated average cheek color
        self.AVERAGE_CHEEK_COLOR = [0, 0, 0]

        self.blushing_occurred_counter = 0

    def detect(self, frame, gray_frame, face_region):
        # extract the right and left cheek coordinates, then use the
        # coordinates to compute the average cheeks color
        right_cheek = face_region[self.right_cheek_idx]
        left_cheek = face_region[self.left_cheek_idx]

        self.draw_cheeks(frame, right_cheek, left_cheek)

        cheeks_color = self.calculate_cheeks_color(frame, gray_frame, right_cheek, left_cheek)

        # check to see if the blushing occurred, and if so, increment the frame counter
        blushing = self.is_blushing(cheeks_color)
        if blushing:
            self.blushing_frame_counter += 1
            print("BLU " + str(self.blushing_frame_counter))

        # if the blushing continued for a sufficient number of frames
        if self.blushing_frame_counter >= self.BLUSHING_CONSECUTIVE_FRAMES:
            self.print_blushing(frame)
            print("BLUSHING")
            # reset the blushing frame counter
            self.blushing_frame_counter = 0
            self.blushing_occurred_counter += 1

        # if blushing is not detected in the current frame, but is detected in 20 previous ones,
        # the counter shouldn't reset because this may occur due to sudden movement, or lighting change
        elif not blushing and self.blushing_frame_counter < 20:
            # reset the blushing frame counter
            self.blushing_frame_counter = 0

    def is_blushing(self, temp_color):
        # compute the change in red, green and blue aspects between new and average cheek color
        red_change = abs(temp_color[2] - self.AVERAGE_CHEEK_COLOR[2])
        green_change = abs(temp_color[1] - self.AVERAGE_CHEEK_COLOR[1])
        blue_change = abs(temp_color[0] - self.AVERAGE_CHEEK_COLOR[0])

        accumulated_change = red_change + green_change + blue_change

        return accumulated_change > self.RGB_CHANGE_ALLOWANCE \
               and 10 < red_change < self.RED_CHANGE_ALLOWANCE \
               and 0 < green_change < self.GREEN_CHANGE_ALLOWANCE \
               and 0 < blue_change < self.BLUE_CHANGE_ALLOWANCE

    def set_average_cheek_color(self, average_cheek_color):
        # calculated average cheek color
        self.AVERAGE_CHEEK_COLOR = average_cheek_color

    def get_number_of_blushing_occurred_and_reset(self):
        retVal = self.blushing_occurred_counter
        self.blushing_occurred_counter = 0
        self.blushing_frame_counter = 0
        return retVal

    @staticmethod
    def calculate_cheeks_color(frame, gray_frame, right_cheek, left_cheek):
        extracted_cheeks_frame = numpy.zeros(frame.shape, numpy.uint8)

        # crate mask for calculating right cheek color
        mask = numpy.zeros(gray_frame.shape, numpy.uint8)
        cv2.drawContours(mask, [right_cheek], -1, 255, -1)
        # calculate average color
        right_cheek_color = cv2.mean(frame, mask)
        cv2.drawContours(extracted_cheeks_frame, [right_cheek], -1, right_cheek_color, -1)

        # crate mask for calculating left cheek color
        mask = numpy.zeros(gray_frame.shape, numpy.uint8)
        cv2.drawContours(mask, [left_cheek], -1, 255, -1)
        # calculate average color
        leftCheekColor = cv2.mean(frame, mask)
        cv2.drawContours(extracted_cheeks_frame, [left_cheek], -1, leftCheekColor, -1)

        average_cheek_color = [(leftCheekColor[0] + right_cheek_color[0]) / 2,
                               (leftCheekColor[1] + right_cheek_color[1]) / 2,
                               (leftCheekColor[2] + right_cheek_color[2]) / 2]

        cv2.putText(extracted_cheeks_frame, "BGR: {:.0f}".format(average_cheek_color[0])
                    + " {:.0f}".format(average_cheek_color[1])
                    + " {:.0f}".format(average_cheek_color[2]), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Cheeks', extracted_cheeks_frame)

        return average_cheek_color

    @staticmethod
    def draw_cheeks(frame, left_cheek, right_cheek):
        cv2.drawContours(frame, [right_cheek], -1, (255, 0, 0))
        cv2.drawContours(frame, [left_cheek], -1, (200, 0, 0))

    @staticmethod
    def print_blushing(frame):
        cv2.putText(frame, "BLUSHING", (150, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
