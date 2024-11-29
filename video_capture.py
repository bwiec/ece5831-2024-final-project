import cv2

class video_capture:
    def __init__(self, file):
        self.file = file
        self.vidcap = cv2.VideoCapture(file)

    def get_next_frame(self):
        success, image = self.vidcap.read()
        return success, image