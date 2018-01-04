import cv2
from imutils import face_utils
from scipy.spatial import distance

from main import EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES, EYE_ASPECT_RATIO_THRESHOLD


class BlinkDetector:
    # grab the indexes of the facial landmarks for the left and right eye
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def __init__(self):
        # number of blinks per frame
        self.frame_blink_counter = 0
        # total number of blinks
        self.total_blink_counter = 0

    def detect_blinks(self, frame, shape):
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        left_eye = shape[self.left_eye_start:self.left_eye_end]
        right_eye = shape[self.right_eye_start:self.right_eye_end]
        left_EAR = self.eye_aspect_ratio(left_eye)
        right_EAR = self.eye_aspect_ratio(right_eye)
        EAR = (left_EAR + right_EAR) / 2.0

        self.draw_eyes(frame, left_eye, right_eye)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if EAR < EYE_ASPECT_RATIO_THRESHOLD:
            self.frame_blink_counter += 1

        else:
            # if the eyes were closed for a sufficient number of frames
            # then increment the total number of blinks
            if self.frame_blink_counter >= EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES:
                self.total_blink_counter += 1

            # reset the eye frame counter
            self.frame_blink_counter = 0

            self.print_blinks(frame, self.total_blink_counter, EAR)

    @staticmethod
    def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        eye_height_1 = distance.euclidean(eye[1], eye[5])
        eye_height_2 = distance.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        eye_width = distance.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        EAR = (eye_height_1 + eye_height_2) / (2.0 * eye_width)

        return EAR

    @staticmethod
    def draw_eyes(frame, left_eye, right_eye):
        # compute the convex hull for the left and right eye and visualize each of the eyes
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    @staticmethod
    def print_blinks(frame, blinks, EAR):
        # print the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.4f}".format(EAR), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
