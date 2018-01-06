import cv2
from imutils import face_utils
from scipy.spatial import distance


class PursedLipsDetector:
    # indexes of the facial landmarks for the mouth
    (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def __init__(self):
        # number per frame
        self.frame_pursed_counter = 0
        # total number of detected pursed lips
        self.total_pursed_counter = 0
        # lips aspect ratio to indicate pursing (if the EAR falls below this value)
        self.LIPS_ASPECT_RATIO_THRESHOLD = -1
        # the number of consecutive frames the lips must be below the threshold
        self.PURSED_LIPS_CONSECUTIVE_FRAMES = -1

    def detect(self, frame, face_region):
        # calculate mouth aspect ratio
        LAR, mouth = self.lips_aspect_ratio(face_region, consider_smile=True)
        self.draw_mouth(frame, mouth)

        # check to see if the mouth aspect ratio is below the threshold, and if so,
        # increment the frame counter
        if LAR < self.LIPS_ASPECT_RATIO_THRESHOLD:
            self.frame_pursed_counter += 1
        else:
            # if the mouth were pursed for a sufficient number of frames
            # then increment the total number of pursing
            if self.frame_pursed_counter >= self.PURSED_LIPS_CONSECUTIVE_FRAMES:
                self.total_pursed_counter += 1

            # reset the pursed lips frame counter
            self.frame_pursed_counter = 0
            self.print_pursed_lips(frame, self.total_pursed_counter, LAR)

    def calculate_lips_aspect_ratio_threshold(self, lips_aspect_ratio):
        self.LIPS_ASPECT_RATIO_THRESHOLD = lips_aspect_ratio * 0.8
        self.PURSED_LIPS_CONSECUTIVE_FRAMES = 4

    def get_and_reset_number_of_lip_pursing(self):
        retVal = self.total_pursed_counter
        self.total_pursed_counter = 0
        self.frame_pursed_counter = 0
        return retVal

    @staticmethod
    def lips_aspect_ratio(face_region, consider_smile=False):
        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
        mouth = face_region[PursedLipsDetector.mouth_start:PursedLipsDetector.mouth_end]
        # inner_part_lips = face_region[60:68]
        # outer_part_lips = face_region[48:59]

        top_lip1 = distance.euclidean(mouth[2], mouth[13])
        top_lip2 = distance.euclidean(mouth[3], mouth[14])
        top_lip3 = distance.euclidean(mouth[4], mouth[15])

        bottom_lip1 = distance.euclidean(mouth[8], mouth[17])
        bottom_lip2 = distance.euclidean(mouth[9], mouth[18])
        bottom_lip3 = distance.euclidean(mouth[10], mouth[19])

        # distance between middle points of top and bottom lip to detect if the person is smiling
        # (in which case lips appear thinner then they are)
        smile = distance.euclidean(mouth[14], mouth[18])
        if consider_smile and smile > 3:
            return 5, mouth

        mouth_width = distance.euclidean(mouth[0], mouth[6])
        LAR = (top_lip1 + top_lip2 + top_lip3 + bottom_lip1 + bottom_lip2 + bottom_lip3) / (6.0 * mouth_width)
        return LAR, mouth

    @staticmethod
    def draw_mouth(frame, mouth_contour):

        # inner_part_hull = cv2.convexHull(inner_part_lips)
        # outer_part_hull = cv2.convexHull(outer_part_lips)

        # cv2.drawContours(frame, [inner_part_hull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [outer_part_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [mouth_contour], -1, (255, 255, 51), 1)

    @staticmethod
    def print_pursed_lips(frame, number, MAR=-1):
        # print the total number of pursed lips on the frame along with
        # the computed lips aspect ratio for the frame
        cv2.putText(frame, "Pursed lips: {}".format(number), (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        cv2.putText(frame, "LAR: {:.4f}".format(MAR), (500, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
