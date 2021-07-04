import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W, A, S, D
import pyautogui


# img:      what to draw on
# lines:    what to draw on img
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]),
                        (coords[2],coords[3]), [255,255,255], 3)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # (3.) Modified thresholds
    processed_img = cv2.Canny(processed_img, threshold1=80, threshold2=180)

    # (2.)
    # blurring the image (so points will be closer)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0 )

    vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    processed_img = roi(processed_img, [vertices])

    # (1.)
    # HoughLines = algorithm finding some lanes in a picture
    # more info: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                                 thres, min, max line height
    # threshold:        limit of subset of points for line detection
    # minLineLength:    Line segments shorter than this are rejected.
    # maxLineGap:       Maximum allowed gap between line segments to treat them as single line.
    # returns array of arrays containing lines
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 150, 20, 15)
    
    draw_lines(processed_img, lines)
    
    return processed_img


def main():
    last_time = time.time()
    while(True):
        screen =  np.array(ImageGrab.grab(bbox=(0,40, 800, 640)))
        new_screen = process_img(screen)
        print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', new_screen)
        #cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
