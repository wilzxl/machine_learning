import numpy as np 
import pandas as pd 
from sklearn.svm import SVC,LinearSVC
import preprocess as pr
import time
from collections import Counter
from sklearn.model_selection import GridSearchCV


#get optimal parameters for linear SVM model -- (100, 0.001)
def cal_lsvm_para(train_data):
	param_grid = {'C': [1.0, 10.0, 100.0], 'gamma': [0.001, 0.01, 0.1]}
	param_search = GridSearchCV(SVC(kernel = 'linear'), param_grid, cv = 5, verbose = 3)
	param_search.fit(train_data[0:, 1:], train_data[0:, 0])
	print (param_search.best_score_, param_search.best_params_)

#get optimal parameters for nonlinear SVM model -- (5000, 0.0001)
def cal_nlsvm_para(train_data):
	param_grid = {'C': [1.0, 10.0, 100.0, 1000.0, 5000.0], 'gamma': [0.0001, 0.001, 0.01, 0.1]}
	param_search = GridSearchCV(SVC(kernel = 'rbf'), param_grid, cv = 5, verbose = 3)
	param_search.fit(train_data[0:, 1:], train_data[0:, 0])
	print (param_search.best_score_, param_search.best_params_)

#linear SVM
def lsvm_model(train_data):
	#lsvm = SVC(kernel='linear', C = 100.0, gamma = 0.001)
	lsvm = LinearSVC(C=1).fit(train_data[0:,1:], train_data[0:,0])
	return lsvm

#nonlinear SVM
def nlsvm_model(train_data):
	svm = SVC(kernel='rbf', C = 5000.0, gamma = 0.0001)
	svm = svm.fit(train_data[0:,1:], train_data[0:,0])
	return svm

def svm_classifier(train_data, test_data):
	test_label = np.hstack(pd.DataFrame.from_csv('gender_submission.csv').values)
	print "Begin to train linear model..."
	start_time = time.time()
	l_model = lsvm_model(train_data)
	
	print "Predicting for linear SVM..."
	l_test_prediction = l_model.predict(test_data[:,:])

	print "--- %s seconds ---" % (time.time() - start_time)
	print "Prediction Accuracy for Linear SVM:", np.sum(test_label == l_test_prediction)*1.0/len(test_label)

	print "--------------------------------------------------------------------------"
	print "Begin to train nonlinear model..."
	start_time = time.time()
	nl_model = nlsvm_model(train_data)

	print "Predicting for nonlinear SVM..."
	nl_test_prediction = nl_model.predict(test_data[:,:])

	print "--- %s seconds ---" % (time.time() - start_time)
	print "Prediction Accuracy for Nonlinear SVM:", np.sum(test_label == nl_test_prediction)*1.0/len(test_label)

def cross_validation():
	train_path = pr.process_data('train.csv')
	train_df = pd.DataFrame.from_csv(train_path)
	train_data = train_df.values
	test_path = pr.process_data('test.csv')
	test_df = pd.DataFrame.from_csv(test_path)
	test_data = test_df.values
	svm_classifier(train_data, test_data)


# For test & debug
if __name__ == '__main__':
	cross_validation()


"""
   From results we find linear SVM performs better than nonlinear. Why?
   1. Maybe I didn't find the best parameters for nlsvm;
   2. The features given are much better than I thought, and good enough to evaluate accurate results, especially 
   for binary class classification problem.
   3. The data distribution is linearly separable.

"""