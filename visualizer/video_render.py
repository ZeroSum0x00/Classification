import cv2
import numpy as np



class VideoRender:
    def __init__(self, frame_duration=1, save_path="out.mp4"):
        self.video_path = save_path
        self.capture = None
        self.vout = None
        self.fps = 30
        self.num_frames_per_image = self.fps * frame_duration

    def _init_video_writer(self, fps, size):
        self.height, self.width, _ = size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (self.width, self.height))
        return video_writer

    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = cv2.imread(image)

        if self.capture == None:
            self.capture = self._init_video_writer(self.fps, image.shape)

        isize = image.shape
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height))

        for _ in range(round(self.num_frames_per_image)):
            self.capture.write(image)
        
    def release(self):
        print("release")
        self.capture.release()
        cv2.destroyAllWindows()
        