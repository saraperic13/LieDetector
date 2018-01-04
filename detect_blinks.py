import time

import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FILE_VIDEO_STREAM_PATH = ""

# eye aspect ratio to indicate blink (if the EAR falls below this value)
EYE_ASPECT_RATIO_THRESHOLD = 0.19
# the number of consecutive frames the eye must be below the threshold
EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES = 2


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


def draw_eyes(frame, left_eye, right_eye):
    # compute the convex hull for the left and right eye and visualize each of the eyes
    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)


def print_blinks(frame, blinks, EAR):
    # print the total number of blinks on the frame along with
    # the computed eye aspect ratio for the frame
    cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR: {:.4f}".format(EAR), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process():
    # initialize the frame counters and the total number of blinks
    frame_blink_counter = 0
    total_blink_counter = 0

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if file_stream and not video_stream.more():
            break

        # get the frame from the threaded video file stream, resize it, and convert it to grayscale
        frame = video_stream.read()
        frame = imutils.resize(frame, width=450)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        faces = detector(gray_frame, 0)

        if faces and faces[0]:
            face = faces[0]
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray_frame, face)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            left_eye = shape[left_eye_start:left_eye_end]
            right_eye = shape[right_eye_start:right_eye_end]
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            EAR = (left_EAR + right_EAR) / 2.0

            draw_eyes(frame, left_eye, right_eye)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if EAR < EYE_ASPECT_RATIO_THRESHOLD:
                frame_blink_counter += 1

            else:
                # if the eyes were closed for a sufficient number of frames
                # then increment the total number of blinks
                if frame_blink_counter >= EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES:
                    total_blink_counter += 1

                # reset the eye frame counter
                frame_blink_counter = 0

                print_blinks(frame, total_blink_counter, EAR)

        # show the frame
        cv2.imshow("Blinks detector", frame)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


if __name__ == "__main__":
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    # vs = FileVideoStream(FILE_VIDEO_STREAM_PATH).start()
    # fileStream = True
    video_stream = VideoStream(src=0).start()
    file_stream = False
    time.sleep(1.0)

    process()

    cv2.destroyAllWindows()
    video_stream.stop()
