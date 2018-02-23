import time

import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream, FileVideoStream

import BlinkDetector
import BlushingDetector
import Person
import PursedLipsDetector
import kNN

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FILE_VIDEO_STREAM_PATH = "../dataset/03.mp4"
DATASET_PATH = 'files/datasetExtracted.csv'

NUMBER_OF_FRAMES_TO_INSPECT = 100
NUMBER_OF_FRAMES_TO_INSPECT_EYES = 25


class LieDetector:

    def __init__(self):
        self.initialize()
        self.frame_counter = 0
        self.blink_detector = BlinkDetector.BlinkDetector()
        self.pursed_lips_detector = PursedLipsDetector.PursedLipsDetector()
        self.blushing_detector = BlushingDetector.BlushingDetector()
        self.person = Person.Person()
        self.questions_counter = 1
        self.seconds = 0

    def initialize(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

        # self.video_stream = FileVideoStream(FILE_VIDEO_STREAM_PATH).start()
        # self.file_stream = True
        self.video_stream = VideoStream(src=0).start()
        self.file_stream = False
        time.sleep(1.0)

    def process(self):

        timeBefore = time.time()
        # loop over frames from the video stream
        while True:

            # if this is a file video stream, check if there are any more frames left in the buffer to process
            if self.file_stream and not self.video_stream.more():
                break

            # get the frame from the threaded video file stream, resize it, and convert it to grayscale
            frame = self.video_stream.read()
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.frame_counter += 1

            # detect faces in the grayscale frame
            faces = self.detector(gray_frame, 0)

            if faces and faces[0]:
                face = faces[0]
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy array
                face_region = self.predictor(gray_frame, face)
                face_region = face_utils.shape_to_np(face_region)

                # inspect face and calculate average values of interest
                if self.frame_counter < NUMBER_OF_FRAMES_TO_INSPECT:

                    if self.frame_counter < NUMBER_OF_FRAMES_TO_INSPECT_EYES:
                        # calculate average eye and lips aspect ratio in the first couple of frames
                        self.calculate_eye_aspect_ratio(face_region)
                        self.calculate_lips_aspect_ratio(face_region)

                    elif self.frame_counter < NUMBER_OF_FRAMES_TO_INSPECT_EYES + 3:
                        # calculate eye and lips aspect ratio threshold value
                        # depending on which blink detector will detect blinks
                        self.blink_detector.calculate_eye_aspect_ratio_threshold(self.person.eye_aspect_ratio)
                        self.pursed_lips_detector.calculate_lips_aspect_ratio_threshold(self.person.lips_aspect_ratio)
                    else:
                        # calculate average number of blinks and lip pursing
                        self.blink_detector.detect(frame, face_region)
                        self.pursed_lips_detector.detect(frame, face_region)

                    # calculate average cheek color
                    self.calculate_average_cheek_color(frame, gray_frame, face_region)

                elif self.frame_counter == NUMBER_OF_FRAMES_TO_INSPECT:
                    print("SET AVERAGE VALUES")
                    # set values of interest to the respective detectors
                    self.blushing_detector.set_average_cheek_color(self.person.average_cheek_color)

                    now = time.time()
                    # set average number of blinks and lip pursing to the person
                    self.person.set_average_number_of_blinks(self.blink_detector.get_and_reset_number_of_blinks(), now-timeBefore)
                    self.person.set_average_number_of_lip_pursing(
                        self.pursed_lips_detector.get_and_reset_number_of_lip_pursing())
                    print(self.person.average_cheek_color)
                    print(self.person.average_number_of_blinks)
                    print(self.person.average_number_of_lip_pursing)

                # detect blinks, lip pursing and blushing
                else:
                    self.blink_detector.detect(frame, face_region)
                    self.pursed_lips_detector.detect(frame, face_region)
                    self.blushing_detector.detect(frame, gray_frame, face_region)

                cv2.putText(frame, "A_EAR: {:.4f}".format(self.person.eye_aspect_ratio), (200, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "A_LAR: {:.4f}".format(self.person.lips_aspect_ratio), (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Lie detector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("x"):
                break

            elif key == ord("n"):

                # calculate number of seconds
                now = time.time()
                self.seconds = now - timeBefore
                self.detect_if_lie()

                self.questions_counter += 1
                timeBefore = time.time()

        now = time.time()
        self.seconds = now - timeBefore
        self.detect_if_lie()

    def calculate_average_cheek_color(self, frame, gray_frame, face_region):
        left_cheek = face_region[self.blushing_detector.left_cheek_idx]
        right_cheek = face_region[self.blushing_detector.right_cheek_idx]
        calculated_cheek_color = self.blushing_detector.calculate_cheeks_color(frame, gray_frame, right_cheek,
                                                                               left_cheek)
        self.person.calculate_average_color(calculated_cheek_color)

    def calculate_eye_aspect_ratio(self, face_region):
        EAR, left_eye, right_eye = self.blink_detector.calculate_eye_aspect_ratio(face_region)
        self.person.calculate_average_eye_aspect_ratio(EAR)

    def calculate_lips_aspect_ratio(self, face_region):
        LAR, mouth = self.pursed_lips_detector.lips_aspect_ratio(face_region, consider_smile=False)
        self.person.calculate_average_lips_aspect_ratio(LAR)

    def detect_if_lie(self):
        # get detected features
        number_of_blinks = self.blink_detector.get_and_reset_number_of_blinks()
        number_of_blushing_occurred = self.blushing_detector.get_number_of_blushing_occurred_and_reset()
        number_of_lip_pursing_occurred = self.pursed_lips_detector.get_and_reset_number_of_lip_pursing()

        if self.seconds > 0:
            number_of_blinks_per_second = number_of_blinks/self.seconds
        else:
            number_of_blinks_per_second = number_of_blinks

        to_predict = [self.person.average_number_of_blinks, number_of_blinks_per_second, number_of_lip_pursing_occurred, number_of_blushing_occurred]
        prediction = kNN.predict([to_predict], DATASET_PATH)

        self.write_to_file(number_of_blinks, number_of_blushing_occurred,
                           number_of_lip_pursing_occurred, number_of_blinks_per_second, prediction)

        # reset seconds counter
        self.seconds = 0

    def write_to_file(self, number_of_blinks, number_of_blushing_occurred, number_of_lip_pursing_occurred,
                      number_of_blinks_per_second, prediction):
        # write report
        file = open("test.txt", "a")
        if self.questions_counter == 1:
            file.write("\n\n******************************************************\n")
            file.write("Person averaged")
            file.write("\n\tblinks: " + str(self.person.average_number_of_blinks))
            file.write("\n\tnumber of blinks per second: " + str(number_of_blinks_per_second))
            file.write("\n\tEAR: " + str(self.person.eye_aspect_ratio))
            file.write("\n\tLAR: " + str(self.person.lips_aspect_ratio))
            file.write("\n\tlip pursing: " + str(number_of_lip_pursing_occurred))
            file.write("\n\tcheek color: " + "{:0.0f}".format(self.person.average_cheek_color[2]) + ", "
                       + "{:0.0f}".format(self.person.average_cheek_color[1]) + ", "
                       + "{:0.0f}".format(self.person.average_cheek_color[0]))

        file.write("\n\n" + str(self.questions_counter) + ". Detected:")
        file.write("\n\tblinks detected: " + str(number_of_blinks))
        file.write("\n\tnumber of blinks per second: " + str(number_of_blinks_per_second))
        file.write("\n\tnumber of blushing occurred:  " + str(number_of_blushing_occurred))
        file.write("\n\tnumber of pursing occurred:  " + str(number_of_lip_pursing_occurred))
        file.write("\n\n\tPredicted:  " + prediction[0])

        file.close()

    def destroy(self):
        cv2.destroyAllWindows()
        self.video_stream.stop()
