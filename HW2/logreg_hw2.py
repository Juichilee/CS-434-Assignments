import numpy as np
import math
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=0.0001
# step_size=1
# step_size=0.1
# step_size=0.01
# step_size=0.00001
# step_size=50
max_iters=1000

def main():

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()
  X_train_bias = dummyAugment(X_train)

  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  # X_train_bias = dummyAugment(X_train)
 
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))


  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  # logging.info("\n---------------------------------------------------------------------------\n")

  # logging.info("Running cross-fold validation for bias case:")

  step_sizes_list = [1., 0.1, 0.01, 0.001, 0.0001]
  best_step = -1
  best_avg_acc = -1
  best_avg_std = 1

  # Perform k-fold cross
    
  for curr_step in step_sizes_list:
        
    global step_size
    step_size = curr_step
    logging.info("\nCurr_Step: {}".format(step_size))

    avg_acc = 0
    avg_std = 0

    for k in [50]:
    # print("X TRAIN BIAS: ", X_train_bias)
      cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
      
      avg_acc += cv_acc
      avg_std += cv_std

      p_cv_acc = cv_acc*100
      p_cv_std = cv_std*100

      logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,p_cv_acc, p_cv_std))

    # avg_acc = avg_acc/7
    # avg_std = avg_std/7
    logging.info("New AVG Step: {}, Acc: {:.4}%, Std: {:.4}% ".format(curr_step, avg_acc*100, avg_std*100))
    if avg_acc > best_avg_acc and avg_std < best_avg_std:
        best_step = curr_step
        best_avg_acc = avg_acc
        best_avg_std = avg_std

    logging.info("Curr Best Step: {}, Acc: {:.4}%, Std: {:.4}% ".format(best_step, best_avg_acc*100, best_avg_std*100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################

  step_size = best_step

  X_test_aug = dummyAugment(X_test)
  w, losses = trainLogistic(X_train_bias,y_train)
  y_pred_test = X_test_aug@w >= 0

  
  test_out = np.concatenate((np.expand_dims(np.array(range(233),dtype=np.int), axis=1), y_pred_test), axis=1)
  header = np.array([["id", "type"]])
  test_out = np.concatenate((header, test_out))
  np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')
  # raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')



######################################################################
# Q3.1 logistic 
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input: 
#   z --   a n-by-1 vector
#
# Output:
#   logit_z --  a n-by-1 vector where logit_z[i] is the result of 
#               applying the logistic function to z[i]
######################################################################
def logistic(wTx):
  # raise Exception('Student error: You haven\'t implemented the logistic calculation yet.')
  # print("WTX SHAPE: ", wTx.shape)
  # print("WTX: ", wTx)
  wTx_size = np.shape(wTx)[0]
  logit_z = np.zeros((wTx_size, 1))
  # print("logit_z: ", logit_z.shape)

  for i in range(wTx_size):
    z = wTx[i] 
    # print("Z: ", z)
    logit_z[i] = 1/(1 + np.exp(-z))
    # if z > 0:
    #   logit_z[i] = -np.logaddexp(0, -z)
    # else:
    #   logit_z[i] = -np.logaddexp(z, 0) + z
  
  # count = 0
  
  # for z in wTx:
  #   logit_z[count] = -np.logaddexp(0, -z)
  #   count+=1

  # logit_z = z
  return logit_z


######################################################################
# Q3.2 calculateNegativeLogLikelihood 
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   w --    a d-by-1 weight bector
#
# Output:
#   nll --  the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X,y,w):
  num_examples = np.shape(X)[0];
  # print("MATRIX X: ", num_examples)
  # print("label vector y: ", y)
  # print("X: ", np.shape(X))
  # print("weight vector w: ", np.shape(w.T))
  W_T_x_i = X@w
  r_sigma = logistic(W_T_x_i)
  neg_r_sigma = logistic(-W_T_x_i)
  # print("W_T_X_I: ", np.shape(W_T_x_i))

  nll = 0;
  for i in range(num_examples):
    y_i = y[i]
    result = (y_i*np.log(r_sigma[i] + 0.0000001) + (1 - y_i)*np.log(neg_r_sigma[i] + 0.0000001))
    # print("RESULT: ", result)
    nll += result

  # print("NLL Before:", nll.item())
  nll = -nll
  # print("NLL After: ", nll.item())
  # raise Exception('Student error: You haven\'t implemented the negative log likelihood calculation yet.')
  return nll.item()



######################################################################
# Q4 trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of 
# iterations max_iters, and step size step_size -- run max_iters of 
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   max_iters --   the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w --  the d-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################
# def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):
def trainLogistic(X,y, max_iters=max_iters):

    # Initialize our weights with zeros
    # n = np.shape(X)[0]
    w = np.zeros( (X.shape[1],1) )
    
    # n size for X
    
    # print("w shape: ", w.shape)
    # print("X shape: ", X.shape)
    # print("X 0: ", X[0])

    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    print("Step Size: ", step_size)
  
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):
               
      # Todo: Compute the gradient over the dataset and store in w_grad
      # .
      # . Implement equation 9.
      # .
      
      # w_grad = 0
      # for j in range(n):
      #   x_i = X[j]
      #   y_i = y[j]
      #   r_sigma = logistic(W_T_x_i[j])
      #   w_grad += (r_sigma - y_i)*x_i
      # print("x_i: ", X)
      # print("wT: ", w)
      W_T_x_i = X@w
      r_sigma = logistic(W_T_x_i)
      # r_sigma_m_y = r_sigma - y
      # print("r_sigma-y shape: ", r_sigma_m_y.shape)
      w_grad = X.T@(r_sigma - y)
      # print("w_grad shape: ", w_grad.shape)

      # raise Exception('Student error: You haven\'t implemented the gradient calculation for trainLogistic yet.')

      # w_grad = w_grad.reshape(w_grad.shape[0], 1)
      # print("w_grad: ", w_grad)
      # print("requried shape: ", (X.shape[1],1))
      # print("w_grad shape: ", w_grad.shape)

      # This is here to make sure your gradient is the right shape
      # assert(w_grad.shape == (X.shape[1],1))

      # Take the update step in gradient descent
      w = w - step_size*w_grad
      
      # Calculate the negative log-likelihood with the 
      # new weight vector and store it for plotting later
      losses.append(calculateNegativeLogLikelihood(X,y,w))
        
    return w, losses


######################################################################
# Q5 dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X --  a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):
  aug_X = np.insert(X, 0, 1, axis=1)
  # print("X: ", X.shape)
  # print("AUG: ", aug_X.shape)
  # raise Exception('Student error: You haven\'t implemented dummyAugment yet.')
  return aug_X


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
