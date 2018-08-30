# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:25:39 2018

@author: Shrey
"""            
            
import pandas
#import time
from random import shuffle
#from copy import deepcopy
import numpy as np
#import mkl
#import csv
#import sklearn
from sklearn import  linear_model, svm,tree,ensemble
from sklearn.model_selection import cross_val_score
from scipy import sparse
import pickle
def findUniqueValues(values_list,csvwriter):
    unique_values_list = []
    for elem in values_list:
        elem = [value.strip() for value in elem.split('|')]
        for j in elem:
            if j not in unique_values_list:
                unique_values_list = unique_values_list +[j]
                csvwriter.writerow([j])
                print('+1')
            else:
                print('.')
    return unique_values_list

def findUniqueValuesDict(values_list,csvwriter,n,head1,head2):
    unique_values_dict = {}
    for i in range(len(values_list)):
        elem = values_list[i]
        elem = [value.strip() for value in elem.split('|')]
        for j in elem:
            if j not in unique_values_dict.keys():
                unique_values_dict[j] = 1
                print(len(unique_values_dict.keys()),i)
            else:
                unique_values_dict[j] = unique_values_dict[j] + 1
                print('.',i)
        if len(unique_values_dict.keys())>=n:
            break
    for key in unique_values_dict.keys():
        csvwriter.writerow({head1:key,head2:unique_values_dict[key]})
    return unique_values_dict
   
def instanceToList(instance,n_train, n_artists, n_genres,members,songs,\
                   unique_genre_ids,unique_artist_name_list)  :
    instance_list = [0]*(86+n_genres+n_artists)
    other_attributes = {'city':22,'age':1,'gender':1,'registered_via':1,\
                        'registration_init_time':1,'expiration_date':1,\
                        'song_length':1,'language':59,'genre_ids':n_genres,\
                        'artist_name':n_artists,'composer':0,\
                        'lyricist':0,}
    try:
        user_position = members.msno.tolist().index(instance[0])
        song_position = songs.song_id.tolist().index(instance[1])
    except:
        return []
    #Add msno
    
    #Add city 0-21
    try:
        instance_list[members.city[user_position]]=1
    except:
        pass
    #Add age 22
    age = members.bd[user_position]
    if (age==0):
        age=29
    instance_list[22] =age
    
    #Add gender 23
    instance_list[23] = members.gender[user_position]
    
    #Add reg_data 24-26
    instance_list[24] = members.registered_via[user_position]
    instance_list[25] = members.registration_init_time[user_position]
    instance_list[26] = members.expiration_date[user_position]
    #Add song length 27
    instance_list[27] = songs.song_length[song_position]
    #Add song language 28-86
    try:
        lang = songs.language[song_position]
        if (lang>0):
            instance_list[27+int(lang)]=1
    except:
        pass
    #Add genres 87-86+n_genre
    genres = songs.genre_ids[song_position]
    try:
        genres=[int(value.strip()) for value in genres.split('|')]
    except:
        genres =[0] #none set to 0 in list
    for genre in genres:
        try:
            genre_index = ((unique_genre_ids.genre_ids.tolist())[0:n_genres]).index(genre)
            instance_list[86+int(genre_index)+1]=1
        except: 
            pass
    
    #Add artists
    artists = songs.artist_name[song_position]
    try:
        artists=[value.strip() for value in artists.split('|')]
    except:
        artists=[]
    for artist in artists:
        try:
            artist_index = (unique_artist_name_list[0:n_artists]).index(artist)
            instance_list[86+artist_index+1+n_genres]=1
        except:
            pass
    return instance_list
    
def formatDataForSKLearn(train_data, n_train, n_artists, n_genres,members,songs,unique_genre_ids,unique_artist_name_list):
    #n_train = traiing data size 73L
    #n_artists = No of artists to take=> max is 230999
    #n_genres
    X=None
    Y=[]
    i=0
    prev_percentage=0
    for instance in train_data:
        instance_list = instanceToList(instance[0:-1],n_train, n_artists, n_genres,members,songs,unique_genre_ids,unique_artist_name_list)
        if(len(instance_list)==0):
            continue
        X=sparse.vstack([X,instance_list])
        Y=Y+[2*instance[-1]-1]
        i=i+1
        percentage = float(i*100)/float(n_train)
        if (percentage>=prev_percentage+5) or (percentage==100):
            prev_percentage=percentage
            print(str(int(percentage))+'%')
        if i>=n_train:
            break
    return (X,Y)

print('Reading User Data')
members = pandas.read_csv('../Kaggle Data/members.csv')
columns_with_nan = members.columns[members.isnull().any()].tolist()
for column_name in columns_with_nan:
	temp = members[column_name].value_counts().keys().tolist()
	members = members.fillna(0)
mapping_members_gender = {'female':-1,'male':1}
members=members.replace({'gender':mapping_members_gender})

print('Reading Song Data')
songs = pandas.read_csv('../Kaggle Data/songs.csv')
columns_with_nan = songs.columns[songs.isnull().any()].tolist()
for column_name in columns_with_nan:
	temp = songs[column_name].value_counts().keys().tolist()
	songs = songs.fillna('none')
    
print('Reading Training Data')
train_data = pandas.read_csv('../Kaggle Data/train.csv')
train_data = train_data.values.tolist()
shuffle(train_data)

print('Reading genre and artists list')
unique_genre_ids= pandas.read_csv('../Kaggle Data/unique_songs_genre_ids_no.csv')
unique_artist_name= pandas.read_csv('../Kaggle Data/unique_songs_artist_name_no2.csv')

print('Done')

#The following code is to generate the additional files required for the project, uncomment to generate those files

#f=open('../Kaggle Data/unique_songs_artist_name.csv','w+', newline='',encoding='utf-8')
#f.close()
#songs_artist_name_list = deepcopy(songs.artist_name.tolist())
#shuffle(songs_artist_name_list)
#with open('../Kaggle Data/unique_songs_artist_name.csv','w', newline='',encoding='utf-8') as csvfile:
#    unique_writer = csv.writer(csvfile, delimiter=' ')
#    unique_writer.writeheader()
#    unique_songs_artist_name = findUniqueValues(songs_artist_name_list,unique_writer)

#songs_genre_ids_list = deepcopy(songs.genre_ids.tolist())
#shuffle(songs_genre_ids_list)
#f=open('../Kaggle Data/unique_songs_genre_ids.csv','w+', newline='',encoding='utf-8')
#f.close()
#with open('../Kaggle Data/unique_songs_genre_ids.csv','w', newline='',encoding='utf-8') as csvfile:
#    unique_writer = csv.writer(csvfile, delimiter=' ')
#    unique_songs_genre_ids = findUniqueValues(songs_genre_ids_list,unique_writer)

#f=open('../Kaggle Data/unique_songs_genre_ids_no.csv','w+', newline='',encoding='utf-8')
#f.close()
#songs_genre_ids_list = deepcopy(songs.genre_ids.tolist())
#shuffle(songs_genre_ids_list)
#with open('../Kaggle Data/unique_songs_genre_ids_no.csv','w', newline='',encoding='utf-8') as csvfile:
#    unique_writer = csv.DictWriter(csvfile, ['genre_ids','No'])
#    unique_songs_genre_ids_no = findUniqueValuesDict(songs_genre_ids_list,unique_writer,10000000,'genre_ids','No')

n_artists=20000
n_genres = 192

n_train = 200000 

print('Creating test data')
print(n_train)
shuffle(train_data)
(X_test,Y_test) = formatDataForSKLearn(train_data, n_train, n_artists, \
                       n_genres,members,songs,unique_genre_ids,unique_artist_name.artist_name.tolist())

print('Saving')
with open('filename.pickle', 'wb') as handle:
    pickle.dump([X_test,Y_test], handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved - Loading')
with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
print('Loaded')
X_test =b[0]
Y_test = b[1]

print('Perceptron')
model1 = linear_model.Perceptron(penalty=None, alpha=0.0001, \
                                        fit_intercept=True, max_iter=None, \
                                        tol=None, shuffle=True, verbose=0, \
                                        eta0=1.0, n_jobs=1, random_state=0, \
                                        class_weight=None, warm_start=False, \
                                        n_iter=None)
print('SVC')
model2 = svm.SVC()

print('DT')
model3=tree.DecisionTreeClassifier()

print('RF')
model4 = ensemble.RandomForestClassifier()

print('SVC - Linear')
model5 = svm.SVC(C=1.0, kernel='Linear')

print('SVC - C = 10^-2')
model6 = svm.SVC(C = 10^-2)

scores = cross_val_score(model1, X_test,Y_test, cv=20)
print('K fold cross validation result for Perceptron:')
print(sum(scores)/20.0) 
scores = cross_val_score(model2, X_test,Y_test, cv=20)
print('K fold cross validation result for SVM:')
print(sum(scores)/20.0)
scores1 = cross_val_score(model3, X_test,Y_test, cv=20)
print('K fold cross validation result for DT:')
print(np.std(np.array(scores1)))
scores = cross_val_score(model4, X_test,Y_test, cv=20)
print('K fold cross validation result for RF:')
print(sum(scores)/20.0)
scores = cross_val_score(model5, X_test,Y_test, cv=20)
print('K fold cross validation result for SVM - Linear:')
print(sum(scores)/20.0)
scores = cross_val_score(model6, X_test,Y_test, cv=20)
print('K fold cross validation result for SVM - C=10^-2:')
print(sum(scores)/20.0)
