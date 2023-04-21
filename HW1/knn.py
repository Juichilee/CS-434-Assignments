from typing import Sized
import numpy as np
import time

def main():

    #############################################################
    # These first bits are just to help you develop your code
    # and have expected ouputs given. All asserts should pass.
    ############################################################

    # I made up some random 3-dimensional data and some labels for us
    example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],
                                 [ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_y = np.array([[0], [1], [1], [1], [0], [1]])
  
    
    # for k in [1,3,5,7,9,99,999,8000]:
    #     t0 = time.time()

    #     predicted_label = predict(example_train_x, example_train_y, example_train_x, k)
    #     train_acc = compute_accuracy(example_train_y, predicted_label)

    #     #######################################
    #     # TODO Compute 4-fold cross validation accuracy
    #     #######################################
    #     val_acc, val_acc_var = cross_validation(example_train_x, example_train_y, 4, k)

        
    #     t1 = time.time()
    #     print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))

    #########
    # Sanity Check 1: If I query with examples from the training set 
    # and k=1, each point should be its own nearest neighbor

    # for i in range(len(example_train_x)):
    #     assert([i] == get_nearest_neighbors(example_train_x, example_train_x[i], 1))
        
    # #########
    # # Sanity Check 2: See if neighbors are right for some examples (ignoring order)
    # nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 1, 4, 2] ), 2)
    # assert(set(nn_idx).difference(set([4,3]))==set())

    # nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 1, -4, 2] ), 3)
    # assert(set(nn_idx).difference(set([1,0,2]))==set())

    # nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 10, 40, 20] ), 5)
    # assert(set(nn_idx).difference(set([4, 3, 0, 2, 1]))==set())

    # #########
    # # Sanity Check 3: Neighbors for increasing k should be subsets
    # query = np.array( [ 10, 40, 20] )
    # p_nn_idx = get_nearest_neighbors(example_train_x, query, 1)
    # for k in range(2,7):
    #   nn_idx = get_nearest_neighbors(example_train_x, query, k)
    #   assert(set(p_nn_idx).issubset(nn_idx))
    #   p_nn_idx = nn_idx
   
    # #########
    # # Test out our prediction code
    # queries = np.array( [[ 10, 40, 20], [-2, 0, 5], [0,0,0]] )
    # pred = predict(example_train_x, example_train_y, queries, 3)
    # assert( np.all(pred == np.array([[0],[1],[0]])))

    # #########
    # # Test our our accuracy code
    # true_y = np.array([[0],[1],[2],[1],[1],[0]])
    # pred_y = np.array([[5],[1],[0],[0],[1],[0]])                    
    # assert( compute_accuracy(true_y, pred_y) == 3/6)

    # pred_y = np.array([[5],[1],[2],[0],[1],[0]])                    
    # assert( compute_accuracy(true_y, pred_y) == 4/6)


    #######################################
    # Now on to the real data!
    #######################################

    # Load training and test data as numpy matrices 
    train_X, train_y, test_X = load_data()


    #######################################
    # Q9 Hyperparmeter Search
    #######################################

    #Search over possible settings of k
    
    # print("Performing 4-fold cross validation")
    # for k in [1,3,5,7,9,99,999,8000]:
    #   t0 = time.time()

    #   #######################################
    #   # TODO Compute train accuracy using whole set
    #   #######################################
    #   predicted_label = predict(train_X, train_y, train_X, k)
    #   train_acc = compute_accuracy(train_y, predicted_label)

    #   #######################################
    #   # TODO Compute 4-fold cross validation accuracy
    #   #######################################
    #   val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k)
      
    #   t1 = time.time()
    # #   print("k = {:5d} -- train acc = {:.2f}% ".format(k, train_acc*100))
    #   print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))
    
    #######################################


    #######################################
    # Q10 Kaggle Submission
    #######################################

    # k_value = np.shape(train_y)[0]
    # k_value = np.sqrt(k_value)
    # k_value = np.round(k_value)

    # print("K_VALUE: ", k_value)
    # # TODO set your best k value and then run on the test set
    best_k = 99

    # Make predictions on test set
    pred_test_y = predict(train_X, train_y, test_X, best_k)    
    
    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=np.int), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')

######################################################################
# Q7 get_nearest_neighbors 
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the 
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out 
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy. 
#
# Input: 
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################

def get_nearest_neighbors(example_set, query, k):
    #TODO

    num_cols = np.shape(example_set)[1]

    diff_set = example_set - query

    u_dist = np.linalg.norm(diff_set, None, 1)
    num_rows = np.shape(example_set)[0]

    dist_dict = {}
    s_dist_dict = {}

    for row_idx in range(num_rows):
        dist_dict[u_dist[row_idx]] = row_idx
        s_dist_dict[row_idx] = u_dist[row_idx]

    # if num_rows is less than k, set k to num_rows
    if num_rows < k:
        k = num_rows

    
    s_dist = np.argpartition(u_dist, k-1, 0)
    

    idx_of_nearest = s_dist[:k]


    return idx_of_nearest, s_dist_dict


######################################################################
# Q7 knn_classify_point 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################

def knn_classify_point(examples_X, examples_y, query, k):
    #TODO
    idx_of_nearest, dist_dict = get_nearest_neighbors(examples_X, query, k)

    label_list = np.take(examples_y, idx_of_nearest)
    label_list = label_list.astype(int)
    for idx in range(len(idx_of_nearest)):
        label_list[idx] = examples_y[idx_of_nearest[idx]][0]

    zero_weight = 0.0
    one_weight = 0.0

    for idx in range(len(idx_of_nearest)):
        curr_idx = idx_of_nearest[idx]
        idx_dist = dist_dict[curr_idx]
        idx_label = label_list[idx]
        
        inverse_dist = 1.0/idx_dist
        if idx_label == 0:
            zero_weight += inverse_dist
        else:
            one_weight += inverse_dist

    predicted_label = 0
    if zero_weight < one_weight:
        predicted_label = 1

    return predicted_label




######################################################################
# Q8 cross_validation 
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input: 
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################

def cross_validation(train_X, train_y, num_folds=4, k=1):
    #TODO
    x_array = np.array(train_X)
    y_array = np.array(train_y)

    split_x_array = np.array_split(x_array, num_folds)
    split_y_array = np.array_split(y_array, num_folds)
    print("Curr k:", k)

    cum_val_acc = [0]*num_folds
    #Train each row in the fold 
    for i in range(num_folds):
        # Get query and label set

        query_set = split_x_array[i]
        query_label_set = split_y_array[i]
        query_set_size = query_label_set.size

        train_set = np.delete(split_x_array, i, 0)
        train_set = np.vstack(tuple(train_set))
        
        train_label_set = np.delete(split_y_array, i)
        train_label_set = np.vstack(tuple(train_label_set))

        predicted_label_list = predict(train_set, train_label_set, query_set, k)

        set_acc = compute_accuracy(query_label_set, predicted_label_list)
        cum_val_acc[i] = set_acc
    

    avg_val_acc = sum(cum_val_acc)/float(num_folds)

    deviations = [(x - avg_val_acc) ** 2 for x in cum_val_acc]
    varr_val_acc = sum(deviations)/float(num_folds)


    return avg_val_acc, varr_val_acc



##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   true_y --  a n-by-1 vector where each value corresponds to 
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match 
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy

######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries 
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################

def predict(examples_X, examples_y, queries_X, k): 
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]

    return np.array(predicted_y,dtype=np.int)[:,np.newaxis]

# Load data
def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:,np.newaxis]
    
    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X


    
if __name__ == "__main__":
    main()