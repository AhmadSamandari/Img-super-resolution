from super-res import *

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
