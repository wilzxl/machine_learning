import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from collections import Counter
#from sklearn.model_selection import GridSearchCV


def process_data(file_name):
	df = pd.DataFrame.from_csv(file_name)
	#process Name
	df['NameLength'] = df.Name.apply(lambda x:len(x))

	#process Parch & SibSp
	df['FamilySize'] = df.SibSp + df.Parch

	#process Age
	age_mean = df.Age.mean()	
	df['Age'] = df['Age'].fillna(age_mean)

	#process Fare
	class_fare = Counter()
	for i in range(0, 3):
		class_fare[i] = df.Fare[df.Pclass == (i+1)].mean()
	#print class_fare
	for i in range(0, 3):
		df.Fare[df['Pclass'] == (i+1)] = df.Fare[df['Pclass'] == (i+1)].fillna(class_fare[i])

	#process Sex
	df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

	#process Embarkation -- fill NA with most frequent embarkation
	#print df['Embarked'].value_counts()
	df['Embarked'] = df['Embarked'].fillna('S').map({'C':1, 'S':2, 'Q':3})
	# df.append({
	# 				 'fullAge': df['Age'],
	# 				 'fullFare': df['Fare'],
	# 				 'NameLength': df['NameLength']}, ignore_index = True)
	
	df = df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1)
	save_path = 'new_' + file_name
	df.to_csv(save_path)
	return save_path

