import time

import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream

import BlinkDetector
import PursedLipsDetector

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FILE_VIDEO_STREAM_PATH = ""


def process():
    blinkDetector = BlinkDetector.BlinkDetector()
    pursedLipsDetector = PursedLipsDetector.PursedLipsDetector()

    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if file_stream and not video_stream.more():
            break

        # get the frame from the threaded video file stream, resize it, and convert it to grayscale
        frame = video_stream.read()
        frame = imutils.resize(frame, width=650)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        faces = detector(gray_frame, 0)

        if faces and faces[0]:
            face = faces[0]
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray_frame, face)
            shape = face_utils.shape_to_np(shape)

            blinkDetector.detect_blinks(frame, shape)
            pursedLipsDetector.detect(frame, shape)

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

    # video_stream = FileVideoStream(FILE_VIDEO_STREAM_PATH).start()
    # file_stream = True
    video_stream = VideoStream(src=0).start()
    file_stream = False
    time.sleep(1.0)

    process()

    cv2.destroyAllWindows()
    video_stream.stop()
