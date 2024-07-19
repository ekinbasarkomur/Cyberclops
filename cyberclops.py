import numpy as np
import cv2
import datetime
import pytesseract
from PIL import Image

def available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

    
def main():
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
        

    print(available_cameras())

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)

    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_sleep = int(1000 / cap_fps)
    print('* Capture width:', cap_width)
    print('* Capture height:', cap_height)
    print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0

    # main loop: retrieves and displays a frame from the camera
    while (True):
        # blocks until the entire frame is read
        success, img = cap.read()
        frames += 1
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(Image.fromarray(img1))
        print(text)

        # compute fps: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)

        # draw FPS text and display image
        cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("webcam", img)


        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if (key == 27):
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
