import numpy as np
import cv2
import datetime
import pytesseract
from PIL import Image
import pyperclip


min_conf = 80

start_point = None
end_point = None
drawing = False



pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


def overlay_image(image, overlay, position):
    """Overlay the resized image onto the background at the specified position."""
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure the overlay image fits within the background
    if y + h > image.shape[0]:
        h = image.shape[0] - y
    if x + w > image.shape[1]:
        w = image.shape[1] - x

    # Place the overlay image on the background
    image[y:y+h, x:x+w] = overlay[:h, :w]

    return image





def select_region(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the rectangle
        drawing = not drawing
        if drawing:
            start_point = (x, y)

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the end point as the mouse moves
            end_point = (x, y)


def main():
    # create display window
    cv2.namedWindow("Cyberclops", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Cyberclops", select_region)

    cropped_image = None
    cropped = False


    
    #print(available_cameras())

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
        success, frame = cap.read()
        clone = frame.copy()


        if not success:
            break


        # FPS Calculation
        frames += 1
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)
        cv2.putText(clone, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)



        if start_point and end_point:
            # Draw the rectangle on the frame
            cv2.rectangle(clone, start_point, end_point, (0, 255, 0), 2)

            if not drawing and not cropped:
                x1, y1 = start_point
                x2, y2 = end_point

                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
            
                # Extract and display the selected area from the last frame
                cropped_image = frame[y1:y2, x1:x2]

                ret,thresh1 = cv2.threshold(cropped_image,120,255,cv2.THRESH_BINARY)
                text = str(pytesseract.image_to_string(thresh1, config='--psm 6', lang='tur'))

                text = text.replace('\n', ' ')
                text = text.replace('-\n', '')

                pyperclip.copy(text)
                print(text)
                
                #cropped_image = detect_text(cropped_image)
                cropped = True
            
            elif drawing:
                cropped = False
                


        if cropped_image is not None:
            # Resize the overlay image (optional)
            overlay_height = int(cap_height / 5) # Desired height of the overlay image
            overlay_width = int(cropped_image.shape[1] * (overlay_height / cropped_image.shape[0]))
            
            resized_overlay = cv2.resize(cropped_image, (overlay_width, overlay_height))

            # Position for the overlay image (bottom-right corner)
            position = (frame.shape[0] - resized_overlay.shape[0], 
                        0)
            
            """Overlay the resized image onto the background at the specified position."""
            x, y = position
            h, w = resized_overlay.shape[:2]

            clone[x:x+h, y:y+w] = resized_overlay[:h, :w]


        # Display the frame
        cv2.imshow("Cyberclops", clone)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Exit the loop when 'q' is pressed
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
