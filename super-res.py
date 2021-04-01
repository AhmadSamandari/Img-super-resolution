import os
import sys
import time
import gc
import pickle
import math
from random import shuffle

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras



# CONSTANT DEFINITION

# FOLDER TO LOAD DATA FROM
DATA_PATH = ROOT_PATH+'/MyDrive/Colab_Notebooks/Group_2/Data'
CODE_PATH = ROOT_PATH+'/MyDrive/Colab_Notebooks/Group_2/Code/Python'
MODEL_PATH = ROOT_PATH+'/MyDrive/Colab_Notebooks/Group_2/Model'

# SETUP
# Volume size
N_ROWS_start       = 128
N_COLUMNS_start    = 128
N_SLICES           = 64

N_ROWS_end         = 256
N_COLUMNS_end      = 256

N_SLICES           = 64

# Type
VOLUME_TYPE       = 'nii'
VOLUME_TEMPLATE = "{}/Patient%s/volumeCT_{}_{}_{}_%s.{}".format(
    DATA_PATH,
    N_ROWS_start,
    N_COLUMNS_start,
    N_SLICES,
    VOLUME_TYPE
    )

LABEL_TEMPLATE = "{}/Patient%s/volumeCT_{}_{}_{}_%s.{}".format(
    DATA_PATH,
    N_ROWS_end,
    N_COLUMNS_end,
    N_SLICES,
    VOLUME_TYPE
    )

# DATA
# Number of cases
OVERALL_NUMBER_OF_CASES             = 67
AVAILABLE_NUMBER_OF_CASES            = 0
# TRAINING-VALIDATION-TEST PERCENTAGES
TRAINING_PERC_CASES                 = 0.70
VALIDATION_PERC_CASES               = 0.15
TEST_PERC_CASES                     = 1 - TRAINING_PERC_CASES - VALIDATION_PERC_CASES

# MODEL
# Model ID
ModelID                             = '01'
# Concatenation
CONCATENATION_DIRECTION_OF_FEATURES = 4

# OPTIMIZER
# Regularization factor lambda
L2_REG_LAMBDA                       = 0.001
# Maximum nuber of epochs
MAX_EPOCHS                          = 50
# TAU factor in the learning rate function
#(INITIAL_LEARNING_RATE - FINAL_LEARNING_RATE) * 1 / (1 + math.exp((epoch_number - MAX_EPOCHS) / TAU_EPOCHS))) + FINAL_LEARNING_RATE)
TAU_EPOCHS                          = 25
# Size for batch normalization
BATCH_SIZE                          = 4
# Learning rate profile
INITIAL_LEARNING_RATE               = 0.001
FINAL_LEARNING_RATE                 = 0.0001


# LOSS FUNCTION FOR 3D IMAGE SCANS
def custom_3D_dice_loss(y_true, y_pred):
  numerator   = tf.scalar_mul(2.0, tf.reduce_sum(tf.multiply(y_true, y_pred), axis = (1, 2, 3)))
  denominator = tf.add(tf.reduce_sum(y_true, axis = (1, 2, 3)), tf.reduce_sum(y_pred, axis = (1, 2, 3)))
  return tf.subtract(1.0, tf.divide(numerator, denominator)) # a single scalar for each data-point in the mini-batch

def custom_cross_correlation_loss(y_true, y_pred):
  numerator = tf.reduce_sum(tf.multiply(y_true, y_pred))
  denominator = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(y_true)),tf.reduce_sum(tf.square(y_pred))))
  return tf.subtract(1.0, tf.divide(numerator, denominator))


def psnr (y_true,y_pred):
  res = tf.image.psnr(y_pred,y_true, 1)
  return res

def SSIM (y_true,y_pred):
  ris = tf.image.ssim(y_true, y_pred, 1)
  return ris


# METRICS FOR EVALUATION:
# IoU(i.e. Interseption over Union):
def IoU(y_true, y_pred):
  y_pred = tf.greater(y_pred, 0.5)
  y_true = tf.cast(y_true, 'bool')
  y_pred = tf.cast(y_pred, 'bool')
  tp = tf.count_nonzero(tf.logical_and(y_true, y_pred))
  fn = tf.count_nonzero(tf.logical_and(tf.logical_xor(y_true, y_pred), y_true))
  fp = tf.count_nonzero(tf.logical_and(tf.logical_xor(y_true, y_pred), y_pred))
  return tp / (tp + fn + fp) # single scalar, already averaged over different instances

def cross_corr(y_true, y_pred):
  numerator = tf.reduce_sum(tf.multiply(y_true, y_pred))
  denominator = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(y_true)),tf.reduce_sum(tf.square(y_pred))))
  return tf.divide(numerator, denominator)

my_loss = custom_cross_correlation_loss
my_metrics = [cross_corr , psnr]



# Possible early stopping and time monitoring:
class custom_early_stopping_and_info(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    global tic
    toc = time.time()
    print('%d'%(((toc - tic) / 60) / 60), 'h', '%.0f'%(((toc - tic) / 60) % 60), 'min spent for training, cumulatively')
    if logs['loss'] < 0.001:
      self.model.stop_training = True
      print('\nNOTE: out of training because overfitting')

# Learning rate program:
def custom_learning_rate_program(epoch_number = 0, current_learning_rate = 0):
  """ 'current_learning_rate' argument not used, just for the sake of completeness """
  return ((INITIAL_LEARNING_RATE - FINAL_LEARNING_RATE) * 1 / (1 + math.exp((epoch_number - MAX_EPOCHS) / TAU_EPOCHS))) + FINAL_LEARNING_RATE

# just to visualize it:
plt.plot(range(0, MAX_EPOCHS), [custom_learning_rate_program(epoch_number = n) for n in range(0, MAX_EPOCHS)])
plt.title('learning rate with epochs:')
plt.show()


early_stopper_and_info = custom_early_stopping_and_info()
learning_rate_scheduler = keras.callbacks.LearningRateScheduler(custom_learning_rate_program, verbose = 0)
my_callbacks_list = [early_stopper_and_info, learning_rate_scheduler]


# Read available data
AVAILABLE_NUMBER_OF_CASES = 0
try:
  del trainVolumes
  del trainLabels
  del validationVolumes
  del validationLabels
  del testVolumes
  del testLabels

except:
  pass
gc.collect()
volumes_list = []
labels_list = []

for index_case in range(1, OVERALL_NUMBER_OF_CASES+1):
  # print('Read current case:', index_case)
  case_id = "{:0>3}".format(index_case)
  volume_path = VOLUME_TEMPLATE % (case_id, case_id)
  label_path = LABEL_TEMPLATE % (case_id, case_id)
  if (os.path.exists(volume_path) and os.path.exists(label_path)):
    AVAILABLE_NUMBER_OF_CASES += 1
    volumes_list.append(volume_path)
    labels_list.append(label_path)
  else:
    print(volume_path)
    print(label_path)

# SPLIT TRAINING AND VALIDATION SETS
TRAINING_NUMBER_OF_CASES      = int(AVAILABLE_NUMBER_OF_CASES * TRAINING_PERC_CASES);
VALIDATION_NUMBER_OF_CASES    = int(AVAILABLE_NUMBER_OF_CASES * VALIDATION_PERC_CASES);
TEST_NUMBER_OF_CASES          = AVAILABLE_NUMBER_OF_CASES - TRAINING_NUMBER_OF_CASES - VALIDATION_NUMBER_OF_CASES;
print("Number of cases for training: " + str(TRAINING_NUMBER_OF_CASES))
print("Number of cases for validation: " + str(VALIDATION_NUMBER_OF_CASES))
print("Number of cases for testing: " + str(TEST_NUMBER_OF_CASES))
# Training set
trainVolumes = np.empty((TRAINING_NUMBER_OF_CASES, N_ROWS_start, N_COLUMNS_start, N_SLICES))
trainLabels = np.empty((TRAINING_NUMBER_OF_CASES, N_ROWS_end, N_COLUMNS_end, N_SLICES))
# Validation set
validationVolumes = np.empty((VALIDATION_NUMBER_OF_CASES, N_ROWS_start, N_COLUMNS_start, N_SLICES))
validationLabels = np.empty((VALIDATION_NUMBER_OF_CASES, N_ROWS_end, N_COLUMNS_end, N_SLICES))
# Test set
testVolumes = np.empty((TEST_NUMBER_OF_CASES, N_ROWS_start, N_COLUMNS_start, N_SLICES))
testLabels = np.empty((TEST_NUMBER_OF_CASES, N_ROWS_end, N_COLUMNS_end, N_SLICES))

count           = 0
countTraining   = 0
countValidation = 0
countTest       = 0
for volume, label in zip(volumes_list, labels_list):
  if countTraining < TRAINING_NUMBER_OF_CASES:
    # get the refs to training set
    volumes = trainVolumes
    labels  = trainLabels
    index = countTraining
    countTraining += 1
  elif countValidation < VALIDATION_NUMBER_OF_CASES:
    # get the refs to validation set
    volumes = validationVolumes
    labels  = validationLabels
    index = countValidation
    countValidation += 1
  else:
    # get the refs to validation set
    volumes = testVolumes
    labels  = testLabels
    index = countTest
    countTest += 1
  temp = nib.load(label) # loading current label...
  temp = temp.get_data()
  temp = np.asarray(temp)
  labels[index, :, :, :] = temp # ...into buffer

  temp = nib.load(volume) # loading corresponding volume...
  temp = temp.get_data()
  temp = np.asarray(temp)
  volumes[index, :, :, :] = temp # ...into buffer

#trainVolumes = trainVolumes.reshape(trainVolumes.shape + (1,)) # necessary to give it as input to model
#validationVolumes = validationVolumes.reshape(validationVolumes.shape + (1,)) # necessary to give it as input to model
#testVolumes = testVolumes.reshape(testVolumes.shape + (1,)) # necessary to give it as input to model



# Training set
trainVolumes2D = np.empty((TRAINING_NUMBER_OF_CASES * 64, N_ROWS_start, N_COLUMNS_start))
trainLabels2D = np.empty((TRAINING_NUMBER_OF_CASES * 64, N_ROWS_end, N_COLUMNS_end))

for case in range(len(trainVolumes)):
  for i in range(0, 64):
    a = (64 * case) + i
    trainVolumes2D[a, :, :] = trainVolumes[case, :, :, i]
    trainLabels2D[a, :, :] = trainLabels[case, :, :, i]

# Validation set
validationVolumes2D = np.empty((VALIDATION_NUMBER_OF_CASES * 64, N_ROWS_start, N_COLUMNS_start))
validationLabels2D = np.empty((VALIDATION_NUMBER_OF_CASES * 64, N_ROWS_end, N_COLUMNS_end))

for case in range(len(validationVolumes)):
  for i in range(0, 64):
    a = (64 * case) + i
    validationVolumes2D[a, :, :] = validationVolumes[case, :, :, i]
    validationLabels2D[a, :, :] = validationLabels[case, :, :, i]

# Training set
testVolumes2D = np.empty((TEST_NUMBER_OF_CASES * 64, N_ROWS_start, N_COLUMNS_start))
testLabels2D = np.empty((TEST_NUMBER_OF_CASES * 64, N_ROWS_end, N_COLUMNS_end))

for case in range(len(testVolumes)):
  for i in range(0, 64):
    a = (64 * case) + i
    testVolumes2D[a, :, :] = testVolumes[case, :, :, i]
    testLabels2D[a, :, :] = testLabels[case, :, :, i]
