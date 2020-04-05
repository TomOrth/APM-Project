#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import glob, os
import re

# ## File paths

# # If you use the pickle file -- Skip to that cell

os.chdir("APM-Project/LeapMotion/Leap_Motion_Data/")

glob_list = []

print("Checkpoint -- Reading in file pairs")

#loop through subject folders and glob
for subject in range(25):
    # Change slashes if on windows to \\
    glob_list.append(sorted(glob.glob(str(subject) + "/[A-Z0-9]*.csv")))
    
#function to flatten glob
flatten = lambda l: [item for sublist in l for item in sublist]
glob_list = flatten(glob_list)

lr_pairs = list(zip(*[iter(glob_list)]*2))

# Makes data frames

print("Checkpoint -- Dataframes")

df_list = []

for pair in lr_pairs:
    df_left = pd.read_csv(pair[0], index_col=None).drop(['Unnamed: 0'], axis = 1)
    df_right = pd.read_csv(pair[1], index_col=None).drop(['Unnamed: 0'], axis = 1)
    
    #rename columns
    df_left = df_left.add_prefix('left')
    df_right = df_right.add_prefix('right')
    
    #merge
    df = pd.merge(df_left, df_right, left_on='leftTime', right_on='rightTime').drop('rightTime', axis = 1)

    #covert fist column to time object
    df['leftTime'] = pd.to_datetime(df['leftTime'].str[:-3], format = '%H:%M:%S.%f')
    
    #difference between rows
    df = df.diff().iloc[1:]
    df['leftTime'] = df['leftTime'].dt.total_seconds()
    
    df.rename(columns={'leftTime':'time'}, inplace=True)
    
    #add sign and subject using regex of file name
    # Change slashes if on windows to \\
    subject_sign = re.split(r'/', re.findall('^[^_]+(?=_)', pair[0])[0])
    df.insert(loc = 0, column = 'Subject', value = subject_sign[0])
    df.insert(loc = 0, column = 'Sign', value = subject_sign[1])
    
    df_list.append(df)

print("Checkpoint -- classification")
# ## Hand Classification


#list of signs included in data set
os.chdir("..")
hands_used  = pd.read_csv("signs_f.csv")
os.chdir("../../output")

one_hand = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Dad', 'Mom', 'Brush', 'Blue', 'Bug', 'Candy', 'Cat', 'Deaf', 'Dog', 'Drink', 'Green', 'Hot', 'Hungry', 'Milk', 'Neutral', 'Orange', 'Pig', 'Please', 'Red', 'Thanks', 'Warm', 'Water', 'Where', 'Why']

two_hand = ['CarDrive', 'Cereal', 'Clothes', 'Coat', 'Cold', 'Come', 'Cost', 'Cry', 'Egg', 'Finish', 'Go', 'Good', 'Happy', 'Hurt', 'More', 'Shoes', 'Socks', 'Stop', 'Store', 'What', 'With', 'Work', 'Yellow', 'Big', 'Small']
all_signs = one_hand+two_hand

class hand_selection:
    
    def __init__(self, drop_left=False):
        self.drop_left = drop_left
        
    def transform(self, df_list, hand_list):
        if not self.drop_left:
            subset = [df for df in df_list if df.Sign.values[0] in hand_list]
        else:
            subset = [df.drop(df.filter(regex='left').columns, axis=1) for df in df_list if df.Sign.values[0] in hand_list]
            
        return subset


# ## Feature Extraction

class extraction:
    def __init__(self, df):
        self.df = df
        self.features = dict()
        
    def label(self):
        self.features['label'] = self.df['Sign'].iloc[0]
        self.df = self.df.iloc[:, 2:]
        
    def mean(self):
        for col in self.df:
            self.features[col + ' mean'] = self.df[col].mean()
            
    def stdev(self):
        for col in self.df:
            self.features[col + ' stdev'] = self.df[col].std()
            
    def extract_features(self):
        self.label()
        self.mean()
        self.stdev()

from sklearn.preprocessing import StandardScaler

def return_features(df_list, hand_list, drop_left):
    scaler = StandardScaler()
    
    feature_list = []
    
    select_class = hand_selection(drop_left)
    frames = select_class.transform(df_list, hand_list)
    
    for df in frames:
        class_obj = extraction(df)
        class_obj.extract_features()
        feature_list.append(class_obj.features)
        
    feat_df = pd.DataFrame(feature_list)
    
    y = feat_df.label
    X = scaler.fit_transform(feat_df.drop(['label'], axis = 1))
    
    return X, y

X_one_hand, y_one_hand = return_features(df_list=df_list,
                                         hand_list=one_hand, 
                                         drop_left=True)

X_two_hand, y_two_hand = return_features(df_list=df_list, 
                                         hand_list=two_hand, 
                                         drop_left=False)
X_all_hand, y_all_hand = return_features(df_list=df_list,
                                         hand_list=all_signs,
                                         drop_left=False)


from random import sample
class LeaveOneOut(object):
    def __init__(self, x, y, size, sign_per_samp):
        self.x = x
        self.y = y
        self.size = size
        self.sign_per_samp = sign_per_samp
        self.idx = 0
    def split_and_increment(self):
        indexs = [x for x in range(self.idx*self.sign_per_samp, (self.idx+1)*self.sign_per_samp)]
        print(type(self.y))
        test_x = self.x[indexs]
        test_y = self.y[indexs]
        train_x = np.delete(self.x, indexs, axis=0)
        train_y = self.y.drop(indexs)
        self.idx += 1
        return train_x, test_x, train_y, test_y
    
class TrainTestSplit(object):
    def __init__(self, x, y, size, sign_per_samp):
        self.x = x
        self.y = y
        self.size = size
        self.sign_per_samp = sign_per_samp
        
    def split(self, train_size):
        train_entries = int(self.size*train_size)
        train_idx = [x for x in range(self.size)]
        
        train_samp = sample(range(self.size), train_entries)
        test_samp = list(set(train_idx)-set(train_samp))
        train_indexes = []
        test_indexes = []
        for ts in train_samp:
            train_indexes.extend([x for x in range(ts*self.sign_per_samp, (ts+1)*self.sign_per_samp)])
            
        for ts in test_samp:
            test_indexes.extend([x for x in range(ts*self.sign_per_samp, (ts+1)*self.sign_per_samp)])
        
        train_x = self.x[train_indexes]
        train_y = self.y.drop(test_indexes)
        test_x = self.x[test_indexes]
        test_y = self.y.drop(train_indexes)
        return train_x, test_x, train_y, test_y

LOO_two = LeaveOneOut(X_two_hand, y_two_hand, 25, len(two_hand))
LOO_one = LeaveOneOut(X_one_hand, y_one_hand, 25, len(one_hand))
LOO_all = LeaveOneOut(X_all_hand, y_all_hand, 25, len(all_signs))

TTS_two = TrainTestSplit(X_two_hand, y_two_hand, 25, len(two_hand))
TTS_one = TrainTestSplit(X_one_hand, y_one_hand, 25, len(one_hand))
TTS_all = TrainTestSplit(X_all_hand, y_all_hand, 25, len(all_signs))


# ## Model Selection
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ### Two Hand


import warnings
warnings.filterwarnings('ignore')

lda_accuracy = []
qda_accuracy = []
knn_accuracy = []
rf_accuracy = []
nb_accuracy = []
svm_accuracy = []
mlp_accuracy = []
LAYER_SIZE = (X_two_hand[0].size,64,len(one_hand)+len(two_hand))

"""
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X_two_hand, 
                                                        y_two_hand,
                                                        stratify=y_two_hand, 
                                                        test_size=0.25)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    lda_accuracy.append(clf.score(X_test,y_test))
    
    clf = SVC(decision_function_shape='ovo', kernel='linear', C=1, gamma=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    svm_accuracy.append(score)
    
    clf = MLPClassifier(solver='adam', alpha=0.0001, activation='tanh', hidden_layer_sizes=LAYER_SIZE)
    clf.fit(X_train, y_train)
    mlp_accuracy.append(clf.score(X_test, y_test))
    print(mlp_accuracy[-1])
"""   
# # Two Hand - Split by subject



import warnings
warnings.filterwarnings('ignore')

lda_accuracy = []
qda_accuracy = []
knn_accuracy = []
rf_accuracy = []
nb_accuracy = []
svm_accuracy = []
mlp_accuracy = []
LAYER_SIZE = (X_two_hand[0].size,64,len(one_hand)+len(two_hand))

for i in range(100):
    X_train, X_test, y_train, y_test = TTS_two.split(0.75)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    lda_accuracy.append(clf.score(X_test,y_test))
    
    clf = SVC(decision_function_shape='ovo', kernel='linear', C=1, gamma=1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    svm_accuracy.append(score)
    
    clf = MLPClassifier(solver='adam', alpha=0.0001, activation='tanh', hidden_layer_sizes=LAYER_SIZE)
    clf.fit(X_train, y_train)
    mlp_accuracy.append(clf.score(X_test, y_test))
    print(mlp_accuracy[-1])

accuracy_list = [lda_accuracy, svm_accuracy, mlp_accuracy]

plt.figure(figsize=(5.5, 5.5))
plt.ylim(0, 1)
plt.boxplot(accuracy_list, labels = ['LDA', 'SVM', 'MLP'])
plt.title('Two-Handed Model Performance')
plt.ylabel('Accuracy')
plt.savefig('plots/two_hand_model_performance.png', dpi = 500)

# # One hand - split by subject

lda_accuracy = []
svm_accuracy = []
mlp_accuracy = []
LAYER_SIZE = (X_two_hand[0].size,64,len(one_hand))

for i in range(100):
    X_train, X_test, y_train, y_test = TTS_one.split(0.75)
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    lda_accuracy.append(clf.score(X_test,y_test))
    
    clf = SVC(decision_function_shape='ovo', kernel='linear', C=1, gamma=1)
    clf.fit(X_train, y_train)
    svm_accuracy.append(clf.score(X_test, y_test))
    
    clf = MLPClassifier(solver='adam', alpha=0.0001, activation='tanh', hidden_layer_sizes=LAYER_SIZE)
    clf.fit(X_train, y_train)
    mlp_accuracy.append(clf.score(X_test, y_test))
    print(mlp_accuracy[-1])

accuracy_list = [lda_accuracy, svm_accuracy, mlp_accuracy]


plt.figure(figsize=(5.5, 5.5))
plt.ylim(0, 1)
plt.boxplot(accuracy_list, labels = ['LDA', 'SVM', 'MLP'])
plt.title('One-Handed Model Performance')
plt.ylabel('Accuracy')
plt.savefig('plots/one_hand_model_performance.png', dpi = 500)
