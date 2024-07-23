import numpy as np
import cv2
import datetime
import pytesseract
from PIL import Image
import pyperclip

#pyperclip.copy('text to be copied')
min_conf = 80

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

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


def detect_text(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_data(rgb, output_type = pytesseract.Output.DICT, lang = "tur")
    #text = pytesseract.image_to_string(Image.fromarray(img1))
    #print(text)


    for i in range(0, len(results["text"])): 
        # We can then extract the bounding box coordinates 
        # of the text region from  the current result 
        x = results["left"][i] 
        y = results["top"][i] 
        w = results["width"][i] 
        h = results["height"][i] 
        
        # We will also extract the OCR text itself along 
        # with the confidence of the text localization 
        text = results["text"][i] 
        conf = int(results["conf"][i]) 
        
        # filter out weak confidence text localizations 
        if conf > min_conf: 
            
            # We will display the confidence and text to 
            # our terminal 
            print("Confidence: {}".format(conf)) 
            print("Text: {}".format(text)) 
            print("") 
            
            # We then strip out non-ASCII text so we can 
            # draw the text on the image We will be using 
            # OpenCV, then draw a bounding box around the 
            # text along with the text itself 
            text = "".join(text).strip() 
            cv2.rectangle(image, 
                        (x, y), 
                        (x + w, y + h), 
                        (0, 0, 255), 2) 
            cv2.putText(image, 
                        text, 
                        (x, y - 10),  
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, (0, 255, 255), 3) 

    return image



def mark_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

    return image


def strech_image(image):

    # Mask label
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.Canny(blur, 30, 150)

    # Find contours
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])

    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)

    cv2.circle(image, left, 8, (0, 50, 255), -1)
    cv2.circle(image, right, 8, (0, 255, 255), -1)
    cv2.circle(image, top, 8, (255, 50, 0), -1)
    cv2.circle(image, bottom, 8, (255, 255, 0), -1)


    bw, bh = [400, 200]
    pts1 = np.float32([top, left, bottom, right])
    pts2 = np.float32([[0, 0], [0, bh-1], [bw-1, bh-1], [bw-1, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, M, (bw, bh))

    warped = image[top[1]:bottom[1], left[0]:right[0]]
    
    return image




def main():
    # create display window
    cv2.namedWindow("Cyberclops", cv2.WINDOW_NORMAL)
    
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
        success, img = cap.read()


        # FPS Calculation
        frames += 1
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)
        cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #img = detect_text(img)

        #img = mark_region(img)

        img = strech_image(img)



        # Display the frame
        cv2.imshow("Cyberclops", img)


        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if (key == 27):
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
