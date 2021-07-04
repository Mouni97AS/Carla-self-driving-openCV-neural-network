
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

import cv2

train_data = np.load('training_data.npy')

print('Original size: ' + str(len(train_data)))

'''
# (0.1.) Displaying train data as image
for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
'''

# (0.2.) Displaying train data as a data table
#           - using pandas DataFrame (transports to tabular data)
#           - printing first 5 rows
#           - counting directions
df = pd.DataFrame(train_data)
# print(df.head(5))
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

# (1.) Shuffling data because for CNN doesnt care for linearity
shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

# (2.) Sorting data according to choice direction
    if choice == [1,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
    else:
        print('no matches')


# (3.) Balancing data: same length
forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights
shuffle(final_data)

print('Balanced size: ' + str(len(final_data)))

np.save('training_data_balanced.npy', final_data)

