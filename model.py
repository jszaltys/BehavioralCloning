import csv
import cv2
import skimage.transform as sktransform
import random
import os
import shutil
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import train_test_split

from keras           import optimizers
from keras.models    import Sequential
from keras.layers    import Flatten , Dense
from keras.layers    import Conv2D, MaxPool2D, Dropout , Lambda
from keras.callbacks import Callback



cameras = ['left', 'center', 'right']
steering_correction = [.25,0, -.25]

#############################################################################
#################### CLASS TO SAVE MODEL AT THE END EVERY EPOCH #############
#############################################################################
class WeightsLogger(Callback):
    def __init__(self, root_path):
        super(WeightsLogger, self).__init__()
        self.weights_root_path = os.path.join(root_path, 'weights/')
        shutil.rmtree(self.weights_root_path, ignore_errors=True)
        os.makedirs(self.weights_root_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(os.path.join(self.weights_root_path, 'model_epoch_{}.h5'.format(epoch + 1)))



#############################################################################
################ FUNCTION TO CROPP AND SHIFT IMAGE ##########################
#############################################################################
def cropp_and_shift(image, v_delta):
    top_offset      = random.uniform(.375 - v_delta, .375 + v_delta)
    bottom_offset   = random.uniform(.125 - v_delta, .125 + v_delta)

    top             = int(top_offset    * image.shape[0])
    bottom          = int(bottom_offset * image.shape[0])
    image           = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

#############################################################################
################ FUNCTION TO RANDOMLY CHANGE BRIGHTNESS #####################
#############################################################################
def brightness(image):
    brightness_alpha = np.random.uniform(0.4,0.8)
    image[:,:,0] = cv2.addWeighted(image[:,:,0],1. - brightness_alpha,brightness_alpha,brightness_alpha,0)
    return image


#############################################################################
################ MAIN FUNCTION WHICH GENERATES DATA DURING TRAINING #########
#############################################################################
def sample_generator(data, is_augumented):
    while True:    
        indices             = np.random.permutation(len(data))                              # Generate batch of random indices
        batch_size          = 100                                                           # Set batch size to 100
        for batch in range(0, len(indices), batch_size):

            batch_indices   = indices[batch:(batch + batch_size)]                                                   
            x               = np.empty([0, 32, 128, 3], dtype=np.float32)
            y               = np.empty([0], dtype=np.float32)
       
            for i in batch_indices:
          
                camera_id   = np.random.randint(len(cameras)) if is_augumented else 1       # Selected random camera           
                image       = cv2.imread(data[i][camera_id])                                # Read proper frame image and get steering angle
                image       = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image       = brightness(image)
                angle       = float(data[i][camera_id+3]) + steering_correction[camera_id]  # Apply streering angle correction if left or right camera was chosen 
                 
                v_delta     = .05 if is_augumented else 0                                   # Randomly shift up and down while preprocessing
                image       =  cropp_and_shift(image,v_delta)                              
                                                    
                x           = np.append(x, [image], axis=0)                                 # Append preprocessed images to batch
                y           = np.append(y, [angle])                                         # Append preprocessed angles to batch
            
            flip_indices    = random.sample(range(x.shape[0]), int(x.shape[0] / 2))         # Flip half on images and angles on random id
            x[flip_indices] = x[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield (x, y)



#############################################################################
################ FUNCTION TO GET LIST OF IMAGES FROM CSV FILE ###############
#############################################################################
def get_list_of_images(path):
    lines = []
    with open(path) as file:
        reader  = csv.reader(file)
        for line in reader:
            lines.append(line)
    return lines



#############################################################################
################ FUNCTION TO SET NETWORK TOPOLOGY ###########################
#############################################################################
def set_model():
    model = Sequential()
    model.add(Lambda(lambda x:x/255.0- 0.5,input_shape = (32,128,3)))
    model.add(Conv2D(16, (3, 3), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='elu'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(20, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

    return model





X_train, X_valid = model_selection.train_test_split(get_list_of_images('data/driving_log.csv'), test_size=.2)

net = set_model()

net.fit_generator(sample_generator(X_train,True),
                  samples_per_epoch=len(X_train),
                  epochs=30,
                  validation_data=sample_generator(X_valid, False),
                  callbacks=[WeightsLogger(root_path='')],
                  nb_val_samples=len(X_valid))
           
