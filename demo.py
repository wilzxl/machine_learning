import feature_analysis as fa
import preprocess as pr
import pandas as pd
import numpy as np
import svm
import knn
import lda
import pca

def feature_analysis():
    fa.analysis_data()

def run_demo():
    train_path = pr.process_data('train.csv')
    train_df = pd.DataFrame.from_csv(train_path)
    train_data = train_df.values
    test_path = pr.process_data('test.csv')
    test_df = pd.DataFrame.from_csv(test_path)
    test_data = test_df.values
    test_label = np.hstack(pd.DataFrame.from_csv('gender_submission.csv').values)

    
    # pca part
    print "Applying PCA:"
    N = 3
    print "PCA for train data"
    train_X = train_data[0:,1:]
    train_y = train_data[0:,0]
    train_X = pca.pca_lda(train_X, train_y, N)
    train_data = np.insert(train_X, 0, train_y, axis=1)
    print "PCA for test data"
    test_X = test_data[0:,0:]
    test_y = test_label
    test_data = pca.pca_lda(test_X, test_y, N)
    print "--------------------------------------------------------------------------"

    

    print "Applying SVM:"
    svm.svm_classifier(train_data, test_data)
    print "--------------------------------------------------------------------------"

    print "Applying KNN:"
    K = 50
    knn.knn_classifier(train_data, test_data, K)
    print "--------------------------------------------------------------------------"

    print "Applying LDA:"
    lda.lda_classifier(train_data, test_data)
    
def main():
    feature_analysis()
    run_demo()   


if __name__ == '__main__':
    main()



