import threading
import queue
import cv2
from deepface import DeepFace
from typing import Union, List, Dict, Tuple, Any
import numpy


class BufferedVideoReader:
    """A Buffered Video Reader

    This class will open up a CV2 VideoCapture
    with the specified name (most likely '0' for
    your local webcam). It then starts a background
    thread to write frames to its queue and will analyze
    them when requested.

    Arguments:
        name (Union[str, int]): The device name, ID, or URL
        detector_backend (str): The DeepFace detector backend
    """

    def __init__(self, name: Union[str, int], detector_backend: str = "opencv"):
        self.detector_backend = detector_backend
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.count = 0
        self.prev = []

    def start(self):
        """Starts the video buffering thread which
        will run for the lifetime of the program
        """
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        """Reads a frame from our video capture and then
        writes it to our queue. Also increments the frame
        count
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.q.put(frame)
            self.count += 1

    def read(self):
        """Reads a frame from the queue or returns
        None if the queue is empty

        Returns:
            frame (): The frame at the front of the queue
            None: If the queue is empty
        """
        try:
            return self.q.get(block=True, timeout=1)
        except queue.Empty:
            return None

    def get(self, prop: int) -> Union[str, int]:
        """Returns a specific property about our video stream, such
        as FPS, Frame Height, Width, etc.

        Arguments:
            prop (int): The integer id of the property

        Returns:
            property Union[str, int]: The property's value
        """
        return self.cap.get(prop)

    def process(
        self,
        frame: numpy.ndarray,
        replace_with: numpy.ndarray = None,
        threshold: float = 0.80,
    ) -> Tuple[numpy.ndarray, List[Dict[str, Any]]]:
        """Processes a frame and performs any facial replacements.

        This method uses DeepFace to find all of the faces in the
        frame. For each face it finds, it will then draw a bounding
        box around that face and replace the facial pixels with the
        pixels from the background image.

        Arguments:
            frame (numpy.ndarray): The frame to be processed
            replace_with (numpy.ndarray): The frame to use for replacements
            threshold (float): Confidence threshold. Results under this will
                be ignored.

        Returns:
            result ([numpy.ndarray, List[Dict[str, Any]]]): The annotated frame
                as well as the result objects from DeepFace
        """

        # 1. Perform the facial extraction
        res = DeepFace.extract_faces(
            frame,
            enforce_detection=False,
            detector_backend=self.detector_backend,
        )

        # 2. For each face that we found...
        for idx, r in enumerate(res):
            region = r["facial_area"]
            confidence = r["confidence"]

            if confidence < threshold:
                continue

            # 3. Draw the bounding box
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(
                frame,
                f"ID: {idx}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
            )

            # 4. Replace the pixels in the analyzed frame with the
            #    pixels from the background frame
            if replace_with is not None:
                # Replace the face pixels with the background image
                frame[y : y + h, x : x + w, 0] = replace_with[y : y + h, x : x + w, 0]
                frame[y : y + h, x : x + w, 1] = replace_with[y : y + h, x : x + w, 1]
                frame[y : y + h, x : x + w, 2] = replace_with[y : y + h, x : x + w, 2]
        return frame, res
