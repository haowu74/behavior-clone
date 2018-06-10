import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []

            for sample in batch_samples:
                for i in range(3):
                    source_path = sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    measurement = float(sample[3])
                    if i == 1:
                        measurement += 0.2
                    if i == 2:
                        measurement -= 0.2
                    measurements.append(measurement)
            # for creating augmented training data
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)

lines = []
batch_size = 32

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(reader.line_num > 1):
            lines.append(line)

train_lines, validate_lines= train_test_split(lines, test_size=0.2)

train_generator = generator(train_lines, batch_size=batch_size)
validate_generator = generator(validate_lines, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Conv2D
#NVIDIA model, using Keras 2.0 API
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60,25), (0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2), activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2), activation='relu'))
model.add(Conv2D(48,(5,5),strides=(2,2), activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, validation_data=validate_generator, steps_per_epoch=len(train_lines)/batch_size, epochs=5, validation_steps=len(validate_lines)/batch_size)

model.save('model.h5')



