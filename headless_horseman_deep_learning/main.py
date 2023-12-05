# For opening, reading, and writing video frames
import cv2
from bufferedvideoreader import BufferedVideoReader


def analyze_video(device=0, show=True, save=True):
    # 1. Create a video capture instance.
    # VideoCapture(0) corresponds to your computers
    # webcam
    cap = BufferedVideoReader(device)

    # Lets grab the frames-per-second (FPS) of the
    # webcam so our output has a similar FPS.
    # Lets also grab the height and width so our
    # output is the same size as the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Now lets create the video writer. We will
    # write our processed frames to this object
    # to create the processed video.
    out = cv2.VideoWriter(
        "outpy.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frame_width, frame_height),
    )

    cv2.namedWindow("Video")

    # Click Space Bar To Set Your Background Image
    background = None

    # 2. Create a video capture instance.
    cap.start()

    while True:
        # Capture frame-by-frame
        frame = cap.read()
        annotated = None

        if frame is None:
            break

        if background is not None:
            frame, res = cap.process(frame, replace_with=background)

        annotated = frame.copy()

        # 3. Allow the user to select a background image
        if background is None:
            label = "Press Space To Capture Background"
            (diff_x, diff_y), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            h, w, _ = frame.shape

            cv2.rectangle(
                annotated,
                ((w - diff_x) // 2, h - 2 * diff_y),
                ((w + diff_x) // 2, h),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                ((w - diff_x) // 2, h - diff_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
            )

        # 4. Show the analyzed frame to the user
        cv2.imshow("Video", annotated)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            return True
        elif k % 256 == 32:
            # SPACE pressed
            background = frame

        if save:
            out.write(annotated)


def main():
    analyze_video()


if __name__ == "__main__":
    main()
