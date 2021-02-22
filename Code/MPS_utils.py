# LIBRARIES
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Softmax

import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

# DATASETS PREPARATION for MINST and FashionMNIST
#####################################################
def dataset_preparation(train_set, train_lab, test_set, test_lab):
  # Function to prepare the images to be given 
  # as input to our MPS algorithm
  #
  # Args
  # train_set: raw training set images
  # train_lab: labels of the training set
  # test_set: raw test set images
  # test_lab: labels of the test set
  #
  #
  # Return
  # fin_train_set: prepared training images (2-dim vectors)
  # one_hot_train: training labels in one hot encoding format
  # fin_test_set: prepared test images (2-dim vectors)
  # one_hot_test: test labels in one hot encoding format

    max_val = np.max(train_set)  # normalize pixels
    min_val = np.min(train_set)
    delt_val = max_val - min_val

    norm_train_set = (train_set - min_val) / delt_val
    norm_test_set = (test_set - min_val) / delt_val
    #################################################

    # flattening images
    reshape_train = list(norm_train_set.reshape(len(norm_train_set), -1))
    reshape_test = list(norm_test_set.reshape(len(norm_test_set), -1))

    def psi(x):
        x = np.array((1 - x, x))    # linear map
        return x

    def mapping_dataset(dataset):
        mapped_dataset = []

        for jj in range(len(dataset)):
            psi_map = list(map(psi, dataset[jj])) 
             # apply map to each pixel
            mapped_dataset.append(psi_map)

        return np.array(mapped_dataset)

    fin_train_set = mapping_dataset(reshape_train)
    fin_test_set = mapping_dataset(reshape_test)

    # commute labels in one hot encoding format
    #########################################
    def one_hot_label(labeled_data):
        n = max(labeled_data) + 1
        one_hot = np.zeros((len(labeled_data), n))

        for ii in range(len(labeled_data)):
            one_hot[ii][labeled_data[ii]] = 1

        return one_hot

    one_hot_train = one_hot_label(train_lab)
    one_hot_test = one_hot_label(test_lab)
    #########################################

    return fin_train_set, one_hot_train, fin_test_set, one_hot_test
#####################################################################

# DATASETS PREPARATION for Chest-X-rays (Pediatric pneumonia detection)
######################################################################
def Pneumonia_img(data_loc):
  # Function return the firs pneumonia virus images in original and reshape size
  # Reshape fixed image size
  img_size = 128

  ##############################################################################
  data = []
  label_data = []      # in a one-hot encoded size
  ##############################################################################
  path = os.path.join(data_loc, 'PNEUMONIA')
  for img in os.listdir(path):

      try:
          img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

          # Reshaped images of size (img_size, img_size)
          resized_arr = cv2.resize(img_arr, (img_size, img_size))
          resized_arr = resized_arr[:,:,0]
          if 'virus' in img:
              break

      except Exception as e:
          print(e)

  return img_arr, resized_arr

def extract_default_img_P_vs_N(data_loc):
  # Function to extract Pneumonia or Normal lung data
  label_folder = ['PNEUMONIA', 'NORMAL']
  # Reshape fixed image size
  img_size = 128

  ##############################################################################
  data = []
  label_data = []      # in a one-hot encoded size
  ##############################################################################

  for label in label_folder:
      path = os.path.join(data_loc, label)
      for img in os.listdir(path):
          try:
              img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

              # Reshaped images of size (img_size, img_size)
              resized_arr = cv2.resize(img_arr, (img_size, img_size))
              resized_arr = resized_arr[:,:,0]

              # Identifying images with one-hot encoding labels
              if label == 'NORMAL':
                  data.append(resized_arr)
                  label_data.append(np.array([1, 0]))
              if label == 'PNEUMONIA':
                  data.append(resized_arr)
                  label_data.append(np.array([0, 1]))

          except Exception as e:
              print(e)

  return np.array(data), np.array(label_data)


def extract_default_img_PB_vs_PV(data_loc):
  # Function to extract Virus pneumonia and Bacterial pneumonia lung data
  # Reshape fixed image size
  img_size = 128

  ##############################################################################
  data = []
  label_data = []      # in a one-hot encoded size
  ##############################################################################

  path = os.path.join(data_loc, 'PNEUMONIA')
  for img in os.listdir(path):
      try:
          img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

          # Reshaped images of size (img_size, img_size)
          resized_arr = cv2.resize(img_arr, (img_size, img_size))
          resized_arr = resized_arr[:,:,0]

          # Identifying images with one-hot encoding labels
          if 'virus' in img:
            data.append(resized_arr)
            label_data.append(np.array([1, 0]))
          if 'bacteria' in img:
            data.append(resized_arr)
            label_data.append(np.array([0, 1]))

      except Exception as e:
          print(e)

  return np.array(data), np.array(label_data)


def X_rays_preprocessing(full_data, full_label):
    # Args
    # full_data: all the available data
    # full_label: all the available label (one-hot encoded)
    #
    # Return
    # x_train: train set
    # y_train: label of the train set
    # x_test: test set
    # y_test: label of the test set

    # Function prepare balaced train and test set
    x_train, x_test, y_train, y_test = train_test_split(full_data, full_label, test_size=0.25, random_state=42,
                                                        shuffle=True)

    # Normalizing dataset
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0

    # Reshaping, flattening
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    def psi(x):
        # function to prepare flatten input array legs
        x = np.array((1 - x, x))
        return np.transpose(x, [1, 2, 0])

    x_train = psi(x_train)
    x_test = psi(x_test)

    return x_train, y_train, x_test, y_test
########################################################################################################################

# Matrix Product State algorithm training
########################################################################################################################
def MPS_training(train_dim, label_one_dim, bond_dim,
                 training_set, training_label, test_set, test_label,
                 batch_size, epochs, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)):
    # MPS_training performs the training of a MPS algorithm through gradient descent
    #
    # ARGS
    # train_dim:
    # label_one_dim: dimension of the onehot encoded labels
    # bond_dim: bond dimension
    # training_set, training_label: dataset to train, labels of the dataset
    # test_set, test_label: dataset to test, labels of the dataset
    # batch_size, epochs, optimizer: model hyperparameters
    #
    # RETURN
    # fit_trend: tensorflow history object with the record of the training

    ##############################################################################
    class MatrixProductState(Layer):
        # MatrixProductState as a keras layer accepts as input one tensor and
        # return another tensor as outputs

        def __init__(self, input_dim, n_labels, bond_dim, dtype=tf.float32):
            super(MatrixProductState, self).__init__()

            # Half of input dimension
            self.half_dim = input_dim // 2
            # Trainable layers
            self.right_side = tf.Variable(self.random_init(self.half_dim, 2, bond_dim),
                                          dtype=dtype, trainable=True)
            self.left_side = tf.Variable(self.random_init(self.half_dim, 2, bond_dim),
                                         dtype=dtype, trainable=True)
            self.label_side = tf.Variable(self.random_init(n_labels, 1, bond_dim)[0],
                                          dtype=dtype, trainable=True)

        @staticmethod
        def random_init(input_dim, d_phys, bond_dim):
            # MPS random initialization
            pos = 0
            std = 1e-3
            random_samples = np.random.normal(loc=pos, scale=std,
                                              size=(d_phys, input_dim, bond_dim, bond_dim))
            x = np.stack(d_phys * input_dim * [np.eye(bond_dim)])
            x = x.reshape((d_phys, input_dim, bond_dim, bond_dim))
            x = x + random_samples
            return x

        @staticmethod
        def contraction(lor_side):
            # MPS contraction
            dim = int(lor_side.shape[0])
            while dim > 1:
                half_dim_it = dim // 2
                old_dim = 2 * half_dim_it
                remaining_side = lor_side[old_dim:]
                # reference to the paper figure
                lor_side = tf.matmul(lor_side[0:old_dim:2], lor_side[1:old_dim:2])
                lor_side = tf.concat([lor_side, remaining_side], axis=0)
                dim = half_dim_it + int(dim % 2 == 1)
            return lor_side[0]

        # Defines the computation from inputs to outputs
        # Expect the instance, defined in the init, plus an inputs
        # The input in the training vector
        def call(self, inputs):
            # Inner product between left side of MPS and left side of input vector
            left_side = tf.einsum("slij,bls->lbij", self.left_side, inputs[:, :self.half_dim, :])
            # Inner product between right side of MPS and right side of input vector
            right_side = tf.einsum("slij,bls->lbij", self.right_side, inputs[:, self.half_dim:, :])
            # MPS contraction
            left_side = self.contraction(left_side)
            right_side = self.contraction(right_side)
            # Layer results
            return tf.einsum("bij,cjk,bki->bc", left_side, self.label_side, right_side)

    ##############################################################################

    # Define Sequential Model
    MPS_model = Sequential()
    # Define Layer
    MPS_model.add(MatrixProductState(input_dim=train_dim, n_labels=label_one_dim, bond_dim=bond_dim))
    # Define activation function of the final layer
    MPS_model.add(Softmax())
    # Compile the MPS_model
    MPS_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    # Fit the MPS_model
    fit_trend = MPS_model.fit(training_set, training_label,
                              batch_size=batch_size, epochs=epochs,
                              verbose=1, validation_data=(test_set, test_label))

    return fit_trend
########################################################################################################################