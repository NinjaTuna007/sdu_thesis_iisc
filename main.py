import pandas as pd
import numpy as np

import scipy
from scipy import optimize
from scipy.optimize import minimize
from sklearn import metrics

## Making Synthetic Data

if True: # make synthetic data

  # want synthetic data to be normal conditioned on any gender and any income bracket
  # want 2 features

  # let us have 3000 male rich, 2000 male poor, 2000 female rich, 3000 female poor [male==1, rich==1]

  tot_num = 50000
  #nums = (np.array([[0.3,0.2],[0.2,0.3]]) * tot_num).astype(int)
  nums = (np.array([[0.45,0.25],[0.05,0.25]]) * tot_num).astype(int)
  
  #print(nums)

  list_data = []
  #list_means = [ [[0,-2.5],[5,3]], [[0,3],[2,5]] ]
  list_means = [ [[-5,0],[-1,-1]], [[5,0],[1,1]] ]

  for i in range(2):
    for j in range(2):
      mean_ij = list_means[i][j]  #np.random.rand(2) 
      cov_ij = np.random.rand(2,2)
      cov_ij = np.matmul(cov_ij, cov_ij.T) # to ensure positive semidefinite
      print("For i,j = (",i,',',j,'), we have mean, cov =', mean_ij, cov_ij)
      
      data = np.random.multivariate_normal(mean_ij, cov_ij, nums[i][j])
      
      col_i = np.ones((nums[i][j],1)) * i
      col_j = np.ones((nums[i][j],1)) * j

      data = np.hstack((data, col_i, col_j))
      #print(data)
      list_data.append(data)
  
  #now, join all, and convert to a pandas dataframe with the right labels
  synth_data = np.vstack(tuple(list_data))
  #print(synth_data.shape)
  
  std_train_df = pd.DataFrame(synth_data, columns = ['f_1','f_2','income-per-year','sex'])
  
  ################## END ######################

## Basics of logistic regression 

def sigmoid(x):
  #print("x", x)
  #print("sig", 1/(1+np.exp(-x)))
  return 1/(1+np.exp(-x))

def forward(params, vals): # [w_1,w_2, ... w_n, b, thresh]
  #print(params)
  return sigmoid(np.matmul(vals, params[:-2]) + params[-2]) 

def prob_to_pred(probs, thresh = 0.5):
  #print(probs)
  probs = np.atleast_1d(probs)
  return np.array([1 if ele >= thresh else 0 for ele in probs]).flatten()

def predictions(params, X_data):
  return prob_to_pred(forward(params, X_data), params[-1])

#print(prob_to_pred([1,0.4,0.5, 0.3]))

def accuracy(Y_data, preds):
  Y_data = Y_data.flatten()# reshape((len(Y_data)))
  preds = preds.flatten()

  #print(Y_data.shape, preds.shape)
  #preds = np.array([0,0,1,1])
  #Y_data = np.array([0,1,1,1])
  #print(float(np.sum([Y_data == preds]))/len(Y_data))

  #print(np.sum([Y_data == preds]))
  return float(np.sum([Y_data == preds]))/len(Y_data)

def solo_objective(var, *params):
  lrg_params = var
  X_data, Y_data, thresh, SIGMA = params
  
  lrg_params = np.append(lrg_params, thresh)
  
  preds = predictions(lrg_params, X_data)
  #print(np.matmul(lrg_params.T , lrg_params))

  return metrics.log_loss(Y_data, preds) + SIGMA * np.matmul(lrg_params.T, lrg_params) 
  #return 1- accuracy(Y_data, preds)

def model_predictions(X_data, male_params, female_params):
  ''' input is a set of X data, we want Y data in the same order as output'''
  preds = []
  
  for index,row in X_data.iterrows():
    x_data = row.drop('sex').to_numpy()
    if row['sex'] == 1:
      preds.append(predictions(male_params, x_data))
    else:
      preds.append(predictions(female_params, x_data))
  #print(preds)
  return np.array(preds)

def acc_loss(data, male_params, female_params):
  Y_data = data[['income-per-year','sex']]
  X_data = data.drop('income-per-year', axis=1)

  X_data_male = X_data[X_data['sex'] == 1]
  X_data_male = X_data_male.drop('sex', axis=1).to_numpy()
  
  Y_data_male = Y_data[Y_data['sex'] == 1]
  Y_data_male = Y_data_male.drop('sex', axis=1).to_numpy().flatten()

  male_preds = predictions(male_params,X_data_male).flatten()
  male = np.sum(Y_data_male != male_preds)

  X_data_female = X_data[X_data['sex'] == 0]
  X_data_female = X_data_female.drop('sex', axis=1).to_numpy()
  
  Y_data_female = Y_data[Y_data['sex'] == 0]
  Y_data_female = Y_data_female.drop('sex', axis=1).to_numpy().flatten()
  
  female_preds = predictions(female_params,X_data_female).flatten()
  female = np.sum(Y_data_female != female_preds)

  return (male+female)/len(Y_data)

def fair_loss_fnr(data, male_params, female_params): # delta FNR | Look for places with Y 1 but pred 0
  # placeholder
  Y_data = data[['income-per-year','sex']]
  X_data = data.drop('income-per-year', axis=1)

  X_data_male = X_data[X_data['sex'] == 1]
  X_data_male = X_data_male.drop('sex', axis=1).to_numpy()
  
  Y_data_male = Y_data[Y_data['sex'] == 1]
  Y_data_male = Y_data_male.drop('sex', axis=1).to_numpy()

  male_preds = prob_to_pred(forward(male_params,X_data_male))

  male_cm = metrics.confusion_matrix(Y_data_male, male_preds)
  male_fnr = male_cm[1][0]/(male_cm[1][0]+male_cm[1][1])


  X_data_female = X_data[X_data['sex'] == 0]
  X_data_female = X_data_female.drop('sex', axis=1).to_numpy()
  
  Y_data_female = Y_data[Y_data['sex'] == 0]
  Y_data_female = Y_data_female.drop('sex', axis=1).to_numpy()
  
  female_preds = prob_to_pred(forward(female_params,X_data_female))

  female_cm = metrics.confusion_matrix(Y_data_female, female_preds)
  female_fnr = female_cm[1][0]/(female_cm[1][0]+female_cm[1][1])

  delta = abs(male_fnr - female_fnr)
  #print(delta)
  return delta, male_fnr, female_fnr


def objective_t(var, *params):
  data, male_params, female_params, ETA, SIGMA, time = params

  if time % 2 == 0: # even time point
    female_params = np.append(female_params,var).flatten()
    return acc_loss(data, male_params, female_params) + ETA * fair_loss_fnr(data, male_params, female_params) * sigmoid(time)
    + SIGMA * np.matmul(female_params.T, female_params)
  else: # odd time point
    male_params = np.append(male_params,var).flatten()
    return acc_loss(data, male_params, female_params) + ETA * fair_loss_fnr(data, male_params, female_params) * sigmoid(time)
    + SIGMA * np.matmul(male_params.T, male_params)
  
def objective_t_even(var, *params):
  data, male_params, female_params, ETA, SIGMA, time = params
  female_params = np.append(female_params,var).flatten()
  return acc_loss(data, male_params, female_params) + ETA * fair_loss_fnr(data, male_params, female_params) + SIGMA * 0
 
def objective_t_odd(var, *params):
  data, male_params, female_params, ETA, SIGMA, time = params
  male_params = np.append(male_params,var).flatten()
  return acc_loss(data, male_params, female_params) + ETA * fair_loss_fnr(data, male_params, female_params) + SIGMA * 0

####################END#######################

## Single variable objective

def find_b1(b_0, male_params, female_params, male_mean, male_cov, female_mean, female_cov):

  w_0 = female_params[:-2].flatten().astype(float)
  w_1 = male_params[:-2].flatten().astype(float)
  
  LHS = np.sqrt( np.matmul(np.matmul(w_0.T, female_cov), w_0) )
  RHS = np.sqrt( np.matmul(np.matmul(w_1.T, male_cov), w_1) )

  b_1 = (RHS/LHS) * (b_0 + np.matmul(w_0.T,female_mean)) - np.matmul(w_1.T,male_mean)

  return b_1

def find_b0(b_1, male_params, female_params, male_mean, male_cov, female_mean, female_cov):
  #return find_b1(b_1,female_params, male_params, female_mean, female_cov, male_mean, male_cov)
  w_0 = female_params[:-2].flatten().astype(float)
  w_1 = male_params[:-2].flatten().astype(float)
  
  LHS = np.sqrt( np.matmul(np.matmul(w_0.T, female_cov), w_0) )
  RHS = np.sqrt( np.matmul(np.matmul(w_1.T, male_cov), w_1) )

  b_0 = (LHS/RHS) * (b_1 + np.matmul(w_1.T,male_mean)) - np.matmul(w_0.T,female_mean)

  return b_0


def obj_new (b_0, *params):

  data, male_params, female_params, male_mean, male_cov, female_mean, female_cov = params

  w_0 = female_params[:-2].astype(float)
  w_1 = male_params[:-2].astype(float)

  b_1 = find_b1(b_0, male_params, female_params, male_mean, male_cov, female_mean, female_cov)

  #print("wow1")
  m_params = np.array(list(w_1) + [b_1, 0.5]).flatten().astype(float)
  #print("wow2")
  f_params = np.array(list(w_0) + [b_0, 0.5]).flatten().astype(float)

  return acc_loss(data, m_params, f_params)
  
  ################ END #######################
  
  ## single var optimization
  
def training(data, male_params, female_params, num_epochs = 500, ETA = 0, SIGMA = 1e-3,  lr = 5e-3):
  acc_list = []
  fair_list = []

  Y_data = data[['income-per-year','sex']]
  X_data = data.drop('income-per-year', axis=1)

  X_data_male = X_data[X_data['sex'] == 1]
  X_data_male = X_data_male.drop('sex', axis=1).to_numpy()

  Y_data_male = Y_data[Y_data['sex'] == 1]
  Y_data_male = Y_data_male.drop('sex', axis=1).to_numpy().flatten()

  X_data_female = X_data[X_data['sex'] == 0]
  X_data_female = X_data_female.drop('sex', axis=1).to_numpy()
  #print(X_data_male.shape)
  
  Y_data_female = Y_data[Y_data['sex'] == 0]
  Y_data_female = Y_data_female.drop('sex', axis=1).to_numpy().flatten()
  

  # pull values from derivations:

  X_data_male_1 = data[data['sex'] == 1]
  X_data_male_1 = X_data_male_1[data['income-per-year'] == 1]
  X_data_male_1 = X_data_male_1.drop('income-per-year', axis=1)
  X_data_male_1 = X_data_male_1.drop('sex', axis=1).to_numpy()

  X_data_female_1 = data[data['sex'] == 0]
  X_data_female_1 = X_data_female_1[data['income-per-year'] == 1]
  X_data_female_1 = X_data_female_1.drop('income-per-year', axis=1)
  X_data_female_1 = X_data_female_1.drop('sex', axis=1).to_numpy()


  male_mean = np.mean(X_data_male_1, axis = 0).flatten().astype(float)
  #print(male_mean.shape)
  male_cov = np.cov(X_data_male_1, rowvar=False).astype(float)
  #print(male_cov.shape)

  female_mean = np.mean(X_data_female_1,axis = 0).flatten().astype(float)
  female_cov = np.cov(X_data_female_1, rowvar=False).astype(float)

  
  ranges = [(-np.inf, np.inf)]*(len(male_params) - 1) #+ [(0,1)]
  #print(ranges)

  # initial networks should be fixed at threshold 0.5

  res = optimize.minimize(solo_objective, male_params[:-1], args=(X_data_male, Y_data_male, 0.5, SIGMA), bounds=ranges)
  ros = res.x
 # print(ros)
  ros = ros/np.linalg.norm(ros)
 # print(ros)
  #new_male_params = res.x
  new_male_params = np.append(ros, 0.5)
  #print (res)
  acc_male = accuracy(Y_data_male, predictions(new_male_params, X_data_male))
  print("Initial male accuracy is:", accuracy(Y_data_male, predictions(new_male_params, X_data_male)))

  res = optimize.minimize(solo_objective, female_params[:-1], args=(X_data_female, Y_data_female, 0.5, SIGMA), bounds=ranges)
  ros = res.x
 # print(ros)
  ros = ros/np.linalg.norm(ros)
 # print(ros)

  #new_female_params = res.x
  new_female_params = np.append(ros, 0.5)
  #print(res)

  print("Initial female accuracy is:", accuracy(Y_data_female, predictions(new_female_params, X_data_female)))

  male_preds = predictions(new_male_params,X_data_male)
  male_cm = metrics.confusion_matrix(Y_data_male, male_preds)
  male_fnr = male_cm[1][0]/(male_cm[1][0]+male_cm[1][1])

  female_preds = predictions(new_female_params,X_data_female)
  female_cm = metrics.confusion_matrix(Y_data_female, female_preds)
  female_fnr = female_cm[1][0]/(female_cm[1][0]+female_cm[1][1])

  print("male confusion:", male_cm)
  print("female confusion:", female_cm)

  print("FNR M|F :", male_fnr, "|", female_fnr)

  print("Initial Acc: ", 1-acc_loss(data, new_male_params, new_female_params))

  res = optimize.minimize(obj_new, female_params[-2], args=(data, male_params, female_params, male_mean, male_cov, female_mean, female_cov))

  b_0 = res.x
  b_1 = find_b1(b_0, male_params, female_params, male_mean, male_cov, female_mean, female_cov)

  m_params = np.array(list(new_male_params[:-2]) + [b_1, 0.5]).flatten().astype(float)
  f_params = np.array(list(new_female_params[:-2]) + [b_0, 0.5]).flatten().astype(float)

  print(m_params)
  print(f_params)

  print("Final Acc: ", 1-acc_loss(data, m_params, f_params))
  
  male_preds = predictions(m_params,X_data_male)
  male_cm = metrics.confusion_matrix(Y_data_male, male_preds)
  male_fnr = male_cm[1][0]/(male_cm[1][0]+male_cm[1][1])

  female_preds = predictions(f_params,X_data_female)
  female_cm = metrics.confusion_matrix(Y_data_female, female_preds)
  female_fnr = female_cm[1][0]/(female_cm[1][0]+female_cm[1][1])

  print("male confusion:", male_cm)
  print("female confusion:", female_cm)

  print("FNR M|F :", male_fnr, "|", female_fnr)


  return new_male_params, new_female_params, acc_list, fair_list


# store information about logistic regressor in np array
n = len(std_train_df.columns) - 2 # exclude label column
male_params = np.random.uniform(-1,1,n+2).T # n for dot product, 1 for bias, 1 for threshold | [w_1, w_2, ... w_n, bias, thresh]
female_params = np.random.uniform(-1,1,n+2).T
'''
male_params = np.zeros(n+2).T # n for dot product, 1 for bias, 1 for threshold | [w_1, w_2, ... w_n, bias, thresh]
female_params = np.zeros(n+2).T
'''
#moment of truth!
best_male_params, best_female_params, acc_list, fair_list = training(std_train_df, male_params, female_params, num_epochs = 500, ETA = 0.1, SIGMA = 5e-4)

################ END #####################

## Bias -Shift Algorithm

def new_train(data, male_params, female_params, num_epochs = 500, ETA = 0, SIGMA = 1e-3,  lr = 5e-3, tol = 1e-3):
  acc_list = []
  fair_list = []
  male_fnr_list = []
  female_fnr_list = []

  Y_data = data[['income-per-year','sex']]
  X_data = data.drop('income-per-year', axis=1)

  X_data_male = X_data[X_data['sex'] == 1]
  X_data_male = X_data_male.drop('sex', axis=1).to_numpy()

  Y_data_male = Y_data[Y_data['sex'] == 1]
  Y_data_male = Y_data_male.drop('sex', axis=1).to_numpy().flatten()

  X_data_female = X_data[X_data['sex'] == 0]
  X_data_female = X_data_female.drop('sex', axis=1).to_numpy()
  
  Y_data_female = Y_data[Y_data['sex'] == 0]
  Y_data_female = Y_data_female.drop('sex', axis=1).to_numpy().flatten()
  
  # first, keep thresholds stationary at 0.5, and train other params individually
  #male_params[-1] = 0.5
  #female_params[-1] = 0.5

  X_data_male_1 = data[data['sex'] == 1]
  X_data_male_1 = X_data_male_1[data['income-per-year'] == 1]
  X_data_male_1 = X_data_male_1.drop('income-per-year', axis=1)
  X_data_male_1 = X_data_male_1.drop('sex', axis=1).to_numpy()

  X_data_female_1 = data[data['sex'] == 0]
  X_data_female_1 = X_data_female_1[data['income-per-year'] == 1]
  X_data_female_1 = X_data_female_1.drop('income-per-year', axis=1)
  X_data_female_1 = X_data_female_1.drop('sex', axis=1).to_numpy()


  male_mean = np.mean(X_data_male_1, axis = 0).flatten().astype(float)
  #print(male_mean.shape)
  male_cov = np.cov(X_data_male_1, rowvar=False).astype(float)
  #print(male_cov.shape)

  female_mean = np.mean(X_data_female_1,axis = 0).flatten().astype(float)
  female_cov = np.cov(X_data_female_1, rowvar=False).astype(float)
  
  ranges = [(-np.inf, np.inf)]*(len(male_params) - 1) #+ [(0,1)]
  # initial networks should be fixed at threshold 0.5

  res = optimize.minimize(solo_objective, male_params[:-1], args=(X_data_male, Y_data_male, 0.5, SIGMA), bounds=ranges)
  ros = res.x
 # print(ros)
  ros = ros/np.linalg.norm(ros)
 # print(ros)
  #new_male_params = res.x
  new_male_params = np.append(ros, 0.5)
  #print (res)
  acc_male = accuracy(Y_data_male, predictions(new_male_params, X_data_male))

  res = optimize.minimize(solo_objective, female_params[:-1], args=(X_data_female, Y_data_female, 0.5, SIGMA), bounds=ranges)
  ros = res.x
 # print(ros)
  ros = ros/np.linalg.norm(ros)
 # print(ros)

  #new_female_params = res.x
  new_female_params = np.append(ros, 0.5)
  #print(res)
  acc_female = accuracy(Y_data_female, predictions(new_female_params, X_data_female))

  if acc_male < 0.85 or acc_female < 0.85:
    print("Discarding this run, initial classifiers not optimal!")
    return False, None, None, None, None, None, None

  print("Initial male accuracy is:", acc_male)
  print("Initial female accuracy is:", acc_female)

  male_preds = predictions(new_male_params,X_data_male)
  male_cm = metrics.confusion_matrix(Y_data_male, male_preds)
  male_fnr = male_cm[1][0]/(male_cm[1][0]+male_cm[1][1])

  female_preds = predictions(new_female_params,X_data_female)
  female_cm = metrics.confusion_matrix(Y_data_female, female_preds)
  female_fnr = female_cm[1][0]/(female_cm[1][0]+female_cm[1][1])

  print("male confusion:", male_cm)
  print("female confusion:", female_cm)

  print("FNR M|F :", male_fnr, "|", female_fnr)

  if abs(male_fnr - female_fnr) < 0.05 : # how to justify this threshold of 0.05?
    print("Discarding this run, initial FNRs already too close!")
    return False, None, None, None, None, None, None

  if male_fnr < 0.005 or female_fnr < 0.005:
    print("Discarding this run, initial fnrs too low!")
    return False, None, None, None, None, None, None


  b_0 = new_female_params[-2]
  b_1 = new_male_params[-2]

  lr = min(b_0, b_1) * 1e-2 # choose learning rate to not change biases by too much
  checkpoint = True

  for time in range(num_epochs): # now, play only with biases

  # alternative: find which one has lower fnr, then optimize for the other in terms of this one
    # b1 ---> b1' st fnrs are close. IN this duration, b0 does not change. 

    #lr = min(b_0, b_1) * 1e-2 # dynamic learning rate

    if checkpoint:

      ver = np.random.randint(0,2)

      if time % 2 == ver: # fair_loss_fnr(data, male_params, female_params)
        # pass
        b_0_opt = find_b0(b_1, male_params, female_params, male_mean, male_cov, female_mean, female_cov)
        diff = min(abs(b_0_opt - b_0), 1) # minima to avoid issues when fnrs are 0 or 1
        # update step
        b_0 += np.sign(b_0_opt - b_0) * diff * lr
      
      else:
        b_1_opt = find_b1(b_0, male_params, female_params, male_mean, male_cov, female_mean, female_cov)
        diff = min(abs(b_1_opt - b_1), 1) # minima to avoid issues when fnrs are 0 or 1
        # update step
        b_1 += np.sign(b_1_opt - b_1) * diff * lr

      new_male_params[-2] = b_1
      new_female_params[-2] = b_0

    # note down accuracy and fairness metrics
    acc_list.append(1 - acc_loss(data, new_male_params, new_female_params))
    fair_loss, male_fnr, female_fnr = fair_loss_fnr(data, new_male_params, new_female_params)
    fair_list.append(fair_loss)   
    male_fnr_list.append(male_fnr)
    female_fnr_list.append(female_fnr)

    if time % 10 == 0:
      print("Training! Current epoch is:", time, " and current accuracy is:", acc_list[-1])
      print("Current biases M/F | ", new_male_params[-2], ', ', new_female_params[-2])
      print("Current FNRs M/F | ", male_fnr, female_fnr)
      print("Current fairness loss is:", fair_list[-1])
    
    if fair_loss < tol:
      checkpoint = False
      '''print("Training! Current epoch is:", time, " and current accuracy is:", acc_list[-1])
      print("Current biases M/F | ", new_male_params[-2], ', ', new_female_params[-2])
      print("Current FNRs M/F | ", male_fnr, female_fnr)
      print("Current fairness loss is:", fair_list[-1])
      break'''
  print("Training Complete!")


  return True, new_male_params, new_female_params, acc_list, fair_list, male_fnr_list, female_fnr_list


############### END ###################


