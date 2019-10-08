import cv2
import time
import threading

class VideoStream:
    # Opens a video with OpenCV from file in a thread
    def __init__(self, src, name="VideoStream", real_time=True):
        """Initialize the video stream from a video

        Args:
            src (str): Video file to process.
            name (str, default='VideoStream'): Name for the thread.
            real_time (bool, default='True'): Defines if the video is going to 
                be read at full speed or adjusted to the original frame rate.

        Attributes:
            name (str, default='VideoStream'): Name for the thread.
            stream (cv2.VideoCapture): Video file stream.
            real_time (bool, default='True'): Defines if the video is going to 
                be read at full speed or adjusted to the original frame rate.
            frame_rate (float): Frame rate of the video.
            grabbed (bool): Tells if the current frame's been correctly read.
            frame (nparray): OpenCV image containing the current frame.
            lock (_thread.lock): Lock to avoid race condition.
            _stop_event (threading.Event): Event used to gently stop the thread.

        """
        self.name = name
        self.stream = cv2.VideoCapture(src)
        self.real_time = real_time
        self.frame_rate = self.stream.get(cv2.CAP_PROP_FPS)
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self):
        # Start the thread to read frames from the video stream with target function update
        threading.Thread(target=self.update, daemon=True, name=self.name).start()
        return self

    def update(self):
        # Continuosly iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                if self.real_time:
                    self.grabbed, self.frame = self.stream.read()
                    # Wait to match the original video frame rate
                    time.sleep(1.0/self.frame_rate)
                else:
                    self.grabbed, self.frame = self.stream.read()
            else:
                return
        self.stop()
    
    def read(self):
        if self.stopped():
            print("Video ended")
        return self.frame

    def stop(self):
        self.lock.acquire()
        self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()

class WebcamVideoStream:
    # Opens a video stream with OpenCV from a wired webcam in a thread
    def __init__(self, src, shape=None, name="WebcamVideoStream"):
        """Initialize the video stream from a video

        Args:
            src (int): ID of the camera to use. From 0 to N.
            name (str, default='WebcamVideoStream'): Name for the thread.

        Attributes:
            name (str, default='WebcamVideoStream'): Name for the thread.
            stream (cv2.VideoCapture): Webcam video stream.
            real_time (bool, default='True'): Defines if the video is going to 
                be read at full speed or adjusted to the original frame rate.
            frame_rate (float): Frame rate of the video.
            grabbed (bool): Tells if the current frame's been correctly read.
            frame (nparray): OpenCV image containing the current frame.
            lock (_thread.lock): Lock to avoid race condition.
            _stop_event (threading.Event): Event used to gently stop the thread.

        """
        self.name = name
        self.stream = cv2.VideoCapture(src)
        self.shape = shape
        if self.shape is not None:
            self.stream.set(3, shape[0])
            self.stream.set(4, shape[1])
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self):
        # Start the thread to read frames from the video stream
        threading.Thread(target=self.update, daemon=True, name=self.name).start()
        return self

    def update(self):
        # Continuosly iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                self.grabbed, self.frame = self.stream.read()
            else:
                return
        self.stopped
    
    def read(self):
        return self.frame

    def stop(self):
        self.lock.acquire()
        self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()
