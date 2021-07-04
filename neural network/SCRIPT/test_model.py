# test_model.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.38

def straight():
    PressKey(W)
    
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    ## we could slow down a bit during turning
    if random.randrange(400) == 1:
        ReleaseKey(W)
    else:
        PressKey(W) #modif
        
    PressKey(A)
         
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    ## we could slow down a bit during turning
    if random.randrange(400) == 1:
        ReleaseKey(W)
    else:
        PressKey(W) #modif
        
    PressKey(D)

    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)

# (1.) Loading model
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0,40,800,640))
            # print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,60))

            # (2.) Predicting from the model
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            # print(prediction)

            # (3.) Turning the car according to prediction
            #       - since it is not a round number we put threshol
            #       - eg.: [0.025, 0.054, 0.91] [left, forw, right]
            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
                print('||')
            elif prediction[0] > turn_thresh:
                left()
                print('<<==')
            elif prediction[2] > turn_thresh:
                right()
                print('==>>')
            else:
                straight()
                print('||')

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
