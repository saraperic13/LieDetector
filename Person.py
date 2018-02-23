class Person:

    def __init__(self):
        self.average_cheek_color = [0, 0, 0]
        self.eye_aspect_ratio = 0
        self.lips_aspect_ratio = 0
        self.average_number_of_blinks = 0
        self.average_number_of_lip_pursing = 0

    def set_average_number_of_blinks(self, num, seconds = 1):
        if num > 0 and seconds > 0:
            self.average_number_of_blinks = num/seconds

    def set_average_cheek_color(self, cheek_color):
        self.average_cheek_color = cheek_color

    def calculate_average_color(self, color):
        if self.average_cheek_color[0] == 0 and self.average_cheek_color[1] == 0 and self.average_cheek_color[2] == 0:
            self.average_cheek_color = color
        else:
            self.average_cheek_color[2] = (self.average_cheek_color[2] + color[2]) / 2.0
            self.average_cheek_color[1] = (self.average_cheek_color[1] + color[1]) / 2.0
            self.average_cheek_color[0] = (self.average_cheek_color[0] + color[0]) / 2.0

    def set_average_number_of_lip_pursing(self, num):
        self.average_number_of_lip_pursing = num

    def calculate_average_eye_aspect_ratio(self, ratio):
        if self.eye_aspect_ratio == 0:
            self.eye_aspect_ratio = ratio
        else:
            self.eye_aspect_ratio = (self.eye_aspect_ratio + ratio)/2.0

    def calculate_average_lips_aspect_ratio(self, ratio):
        if self.lips_aspect_ratio == 0:
            self.lips_aspect_ratio = ratio
        else:
            self.lips_aspect_ratio = (self.lips_aspect_ratio + ratio) / 2.0

