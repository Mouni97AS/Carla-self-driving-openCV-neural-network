
# Collecting training data:
# - input: pixels & pressed keys
# - (output will be: pressed key at given pixels)
# - how to train: by manually driving the car
# - we need ~ 100.000 < ... < 1.000.000 training data ->
#   because we are going to balance it (leave out go straight datas)

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os


# (1.) Convert keys to a ...multi-hot... [A,W,D] boolean values.
def keys_to_output(keys):
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output



# (2.) Getting/initializing the training data from file
file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []




def main():

    for i in list(range(3))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    while(True):

        if not paused:

            # (3.) Grab & invert & resize screen
            # to something a bit more acceptable for a CNN (160x120)
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,60))

            # (4.) Grab -> transform pressed key ->
            #      store it as training data with the pixels ->
            #      save it in every 1000th
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
            
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)

        # (5.) Pause creating training data
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()
