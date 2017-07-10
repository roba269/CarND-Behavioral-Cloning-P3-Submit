import cv2
import csv
import numpy as np

lines = []
with open("../data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# generate data from csv file and image files
# didn't use generator because the data size still fits memory
images = []
measurements = []
for line in lines:
	deg = float(line[3])
	img_center = "../data/IMG/" + line[0].split('/')[-1]
	image = cv2.imread(img_center)
	images.append(image)
	measurements.append(deg)

	#using left and right camera doesn't help much for my case
	
	#correction = 0.15
	#img_left = "../data/IMG/" + line[1].split('/')[-1]
	#image = cv2.imread(img_left)
	#images.append(image)
	#measurements.append(deg + correction)
	#img_right = "../data/IMG/" + line[2].split('/')[-1]
	#image = cv2.imread(img_right)
	#images.append(image)
	#measurements.append(deg - correction)

# left-right flip to generate more data
length = len(images)
for idx in range(length):
	flip = cv2.flip(images[idx], 1)
	images.append(flip)
	measurements.append(-measurements[idx])

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Convolution2D, Lambda, Dropout

model = Sequential()

# normalize the data in a simple way
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# crop the less-informative top and bottom parts
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# the following network architecture is mostly same as Nvidia's paper.
# To avoid overfit, added a dropout layer between conv layers and full-connection layers.
model.add(Convolution2D(24,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu'))
model.add(Convolution2D(36,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu'))
model.add(Convolution2D(48,kernel_size=(5,5),strides=(2,2),padding='valid',activation='relu'))
model.add(Convolution2D(64,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu'))
model.add(Convolution2D(64,kernel_size=(3,3),strides=(2,2),padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# use Adam optimzer so learning rate need not to be explicitly specified
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')

