# train_model.py

import numpy as np
from alexnet import alexnet

# (1.) Setting constants
WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

# (2.) Load train data, split training and testing sets
#       - training data is what we'll fit the neural network with
#       - test data is to validate the results &
#            test the accuracy of the network
train_data = np.load('training_data-balanced.npy')
train = train_data[:-500]
test = train_data[-500:]

# (3.) Creating training & test data arrays
#       - X (image)
#       - Y (choice)
X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

# (4.) Training the CNN
#   - inputset - validation set
#   - snapshot_step: how many steps to show on screen
model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS,
        validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# Saving the weights
model.save(MODEL_NAME)



# tensorboard --logdir=foo:C:/path/to/log
