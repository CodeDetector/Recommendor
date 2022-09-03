# -*- coding: utf-8 -*-

from time import sleep, time
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np  
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import random
import json 

# 0 - Easy , 1 - Medium , 2 - Hard
 
def load_dataset() : 
  df = pd.read_csv('data2.csv').drop('Unnamed: 0' , axis = 1)
  df['Difficulty_Category'] = pd.cut(df['Difficulty'], bins=3, labels=[0 , 1 , 2])

  df['Percentile_Category'] = pd.cut(df['Percentile'], bins=7, labels=range(1, 8))

  df['Number_of_attempts'] = np.random.randint(0, 5000, df.shape[0])


  df['Difficulty'] = df.apply(lambda x: x['Percentile_Category'] if x['Number_of_attempts'] >1000 else x['Difficulty'], axis=1)

  s2 = set() 
  s1 = set() 

  for i in df.itertuples():
      for j in i.Topic_ID.strip('[]').split(', ') : 
        s1.add(j)
      
      for k in i.KP_ID.strip('[]').split(', ')  : 
        s2.add(k[1:-1])

  for i in s1 : 
    df[i] = 0

  for i in s2 : 
    df[i] = 0

  count = 0 ; 
  for i in df.itertuples():
      for j in i.Topic_ID.strip('[]').split(', ') : 
        df.iloc[count , df.columns.get_loc(j)] = 1
      
      for k in i.KP_ID.strip('[]').split(', ') :
        df.iloc[count , df.columns.get_loc(k[1:-1])] = 1
      
      count+=1 

  df['HL_only'].replace({'N' : 0 , 'Y' : 1} , inplace = True)
  df.drop(['KP_ID'  , 'Topic_ID'] , axis = 1, inplace = True)

  df.drop(['Percentile_Category' , 'Number_of_attempts'] , axis = 1 , inplace = True )

  return df 

class Recommendor : 

  def __init__(self) : 
    self.df = load_dataset()

  def __calculateSim(self) :  
    Y = cdist(self.df , self.df, 'correlation')
    self.__sim_matrix = pd.DataFrame(data = Y , index = self.df.index , columns = self.df.index) 

  def getSimMatrix(self) : 
    return self.__sim_matrix  

  def start(self, topic) : 
      self.df = self.df.set_index('QID') 
      self.df = self.df[self.df[topic] == 1]
      self.questions = self.df.index
      self.__calculateSim() 
    
  def predict(self , diff):
      qid = None 
      # print(diff)
      frame = self.df[self.df.Difficulty == diff]
      if frame.shape[0] > 0 : 
          qid = frame.sample(1).index[0]
          series = self.__sim_matrix.loc[ : , qid]
          qid =  int(random.choice(list(series.sort_values(ascending = False).index[0 : 100])))
      return qid

class User(Recommendor) : 

  def __init__(self,details) : 
    Recommendor.__init__(self)
    self.__id = details['id']
    self.__topic = details['topic']
    self.__localHistory = details['history']
    self.__localHistory.append(None) 
    self.__globalRating = details['globalRating']
    self.__currDifficulty = details['difficulty'] 
    self.__Acceptance = details['acceptance']
    self.__currLevel = details['level']
    self.__maxDifficulty = {'1' : 3 , '2' : 5 , '3' : 7} 

  def __del__(self):
        print('Destructor called, User deleted.')
  
  def startTest(self)  : 
    self.start(self.__topic)
    return self.__driver()  

  def __driver(self)  : 
    return self.__getPredictions() 

  def __getPredictions(self) :  
    qid = self.predict(self.__currDifficulty) 
#   Count is used for preventing the prediction function to go into infinite loop
    count  = 1   
#   if qid already in local history then ,call predict() again  
    while qid in self.__localHistory : 
      qid = self.predict(self.__currDifficulty)
      count+=1  

#   If qid in history for 10 times then increase the difficulty a bit( by 1 )
      if count == 10 :  
        self.__currDifficulty = min(self.__currDifficulty + 1 , self.__maxDifficulty[self.__currLevel])
        count = 0 
#   If count goes to 1000 then it means there are no relevent questions left for the user to answer 
      if count == 20 : 
        print("Infinite loop ... Exiting ") 
        return 

    self.__localHistory.append(qid)  

    # print("Question  : " , qid)  
    return qid 


  def setAttempt(self , correct) : 
    self.start(self.__topic)
    if correct : 
      self.__Acceptance[self.__currLevel] += 1 
      if self.__Acceptance[self.__currLevel] > 5 :  
        if self.__CurrLevel < 3 : 
          self.__currLevel +=1 
      self.__currDifficulty = min(self.__currDifficulty + 1 , self.__maxDifficulty[self.__currLevel]) 

    if not correct : 
      self.__Acceptance[self.__currLevel] -=1  

    return self.__getPredictions()

  # Getters/ Setters 
  def getAttributes(self) : 
    return {
      'id' :self.__id,
      'globalRating' : self.__globalRating , 
      'history' : self.__localHistory,
      'difficulty' : self.__currDifficulty , 
      'acceptance' : self.__Acceptance ,
      'level' : self.__currLevel
    }
  def getId(self) : 
    return self.__id 

  def getCurrDifficulty(self) : 
    return self.__currDifficulty
  
  def getGlobal(self)  : 
    return self.__globalRating 

  def getLocalHistory(self) : 
    return self.__localHistory
  
  def setTopic(self , topic) : 
    self.__topic = topic  
  
  def getTopic(self) : 
    return self.__topic

  def getAcceptance(self) : 
    return self.__Acceptance 

  def getLevel(self) : 
    return self.__currLevel  


# TODO:Initialize user , save user attributes as user_id.json 
def initialize_user(user_id , global_rating) :  
    if global_rating <=  3  :
      level = '1' 
    elif global_rating > 3 and global_rating < 6 : 
      level = '2' 
    else : 
      level = '3' 
    att = {
      'id' :user_id,
      'globalRating' : global_rating , 
      'history' : list(),
      'difficulty' : global_rating , 
      'acceptance' : {'1' : 0 , '2' : 0 , '3' : 0 },
      'level' : level
    }
    json_object = json.dumps(att, indent=4)
    with open(f'{user_id}.json', "w") as outfile:
        outfile.write(json_object)

# TODO:open user file , generate question , update user attributes in the file 
def generate_question(user_id , topic , attempt) :

  with open(f'{user_id}.json', 'r') as openfile:
    json_object = json.load(openfile)

  json_object['topic'] = topic  
  user = User(json_object) 
  user.setTopic(topic) 
  qid = user.setAttempt(attempt)

  return qid 

# TODO:open user json , generate 1st question , update and save user attributes
def start_test(user_id , topic) : 
  with open(f'{user_id}.json', 'r') as openfile:
    json_object = json.load(openfile)
  json_object['topic'] = topic
  user = User(json_object) 
  qid = user.startTest() 
  json_object = user.getAttributes() 

  json_object = json.dumps(json_object, indent=4)

  with open(f'{user_id}.json', 'w') as outfile:
    outfile.write(json_object)

  return qid 

user_id = np.ceil(time())
initialize_user(user_id, 2 )
qid = start_test(user_id, '103') 
print(qid)
qid = generate_question(user_id=user_id , topic= '103' , attempt = 1  )
print(qid)
qid = generate_question(user_id=user_id , topic= '103' , attempt = 1  )
print(qid)
qid = generate_question(user_id=user_id , topic= '103' , attempt = 1  )
print(qid)
qid = generate_question(user_id=user_id , topic= '103' , attempt = 1  )
print(qid)
qid = generate_question(user_id=user_id , topic= '103' , attempt = 1  )
print(qid)