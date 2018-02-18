import cv2
from imutils import face_utils
from scipy.spatial import distance


class BlinkDetector:
    # indexes of the facial landmarks for the left and right eye
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def __init__(self):
        # number of blinks per frame
        self.frame_blink_counter = 0
        # total number of blinks
        self.total_blink_counter = 0
        # eye aspect ratio to indicate blink (if the EAR falls below this value)
        self.EYE_ASPECT_RATIO_THRESHOLD = -1
        # the number of consecutive frames the eye must be below the threshold
        self.BLINK_CONSECUTIVE_FRAMES = -1

    def detect(self, frame, face_region):

        EAR, left_eye, right_eye = self.calculate_eye_aspect_ratio(face_region)

        self.draw_eyes(frame, left_eye, right_eye)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if EAR < self.EYE_ASPECT_RATIO_THRESHOLD:
            self.frame_blink_counter += 1

        else:
            # if the eyes were closed for a sufficient number of frames
            # then increment the total number of blinks
            if self.frame_blink_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                self.total_blink_counter += 1

            # reset the eye frame counter
            self.frame_blink_counter = 0

        self.print_blinks(frame, self.total_blink_counter, EAR)

    def calculate_eye_aspect_ratio_threshold(self, eye_aspect_ratio):
        self.EYE_ASPECT_RATIO_THRESHOLD = eye_aspect_ratio * 0.7
        self.BLINK_CONSECUTIVE_FRAMES = 1

    def get_and_reset_number_of_blinks(self):
        retVal = self.total_blink_counter
        self.frame_blink_counter = 0
        self.total_blink_counter = 0
        return retVal

    @staticmethod
    def calculate_eye_aspect_ratio(face_region):
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        left_eye = face_region[BlinkDetector.left_eye_start:BlinkDetector.left_eye_end]
        right_eye = face_region[BlinkDetector.right_eye_start:BlinkDetector.right_eye_end]

        left_EAR = BlinkDetector.eye_aspect_ratio(left_eye)
        right_EAR = BlinkDetector.eye_aspect_ratio(right_eye)
        return (left_EAR + right_EAR) / 2.0, left_eye, right_eye

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
    def print_blinks(frame, blinks, EAR=-1):
        # print the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.4f}".format(EAR), (200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
