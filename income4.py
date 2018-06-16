#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 21:00:02 2018

@author: macuser
"""

# Implementation of Gaussian NB on Census Income dataset

# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

# Data Import

adult_df = pd.read_csv('adult_data.csv',
                       header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

# Handling Missing Data

adult_df.isnull().sum()

for value in ['workclass', 'education',
          'marital_status', 'occupation',
          'relationship','race', 'sex',
          'native_country', 'income']:
    print(value,":", sum(adult_df[value] == '?'))
    
# Data preprocessing
adult_df_rev = adult_df
adult_df_rev.describe(include= 'all')

# Data Imputation
for value in ['workclass', 'education',
          'marital_status', 'occupation',
          'relationship','race', 'sex',
          'native_country', 'income']:
    adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include='all')[value][2]],
                                inplace=True)

# One-Hot Encoder
le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(adult_df.workclass)
education_cat = le.fit_transform(adult_df.education)
marital_cat   = le.fit_transform(adult_df.marital_status)
occupation_cat = le.fit_transform(adult_df.occupation)
relationship_cat = le.fit_transform(adult_df.relationship)
race_cat = le.fit_transform(adult_df.race)
sex_cat = le.fit_transform(adult_df.sex)
native_country_cat = le.fit_transform(adult_df.native_country)

#initialize the encoded categorical columns
adult_df_rev['workclass_cat'] = workclass_cat
adult_df_rev['education_cat'] = education_cat
adult_df_rev['marital_cat'] = marital_cat
adult_df_rev['occupation_cat'] = occupation_cat
adult_df_rev['relationship_cat'] = relationship_cat
adult_df_rev['race_cat'] = race_cat
adult_df_rev['sex_cat'] = sex_cat
adult_df_rev['native_country_cat'] = native_country_cat

#drop the old categorical columns from dataframe
dummy_fields = ['workclass', 'education', 'marital_status', 
                  'occupation', 'relationship', 'race',
                  'sex', 'native_country']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

# Standardization of Data

num_features = ['age', 'workclass_cat', 'fnlwgt', 'education_cat', 'education_num',
                'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
                'sex_cat', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country_cat']
scaled_features = {}
for each in num_features:
    mean, std = adult_df_rev[each].mean(), adult_df_rev[each].std()
    scaled_features[each] = [mean, std]
    adult_df_rev.loc[:, each] = (adult_df_rev[each] - mean)/std
num_features.append('income')
adult_df_rev2 =  adult_df_rev[num_features]   
    
# Data Slicing
    
features = adult_df_rev2.values[:,:14]
target = adult_df_rev2.values[:,14]
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size = 0.33, random_state = 10)

#Gaussian Naive Bayes Implementation
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

print('Accuracy Score') 
print(accuracy_score(target_test, target_pred, normalize = True))
