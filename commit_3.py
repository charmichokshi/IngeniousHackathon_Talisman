# coding: utf-8

# In[1]:


#data dictionary

#list of fields
field = ['Data Science','Software development','Game dvelopment']


# In[2]:


#list of domains
domain = [['Data Analyst','Machine Learning','Computer Vision'],
          ['Web Development(Front End)', 'Android Development','iOS Development'],
         ['Unity Game Development','Android Game Development', 'iOS Game Development']]


# In[22]:


#ideal skills
skills = [
           [{'excel':1,'data_visualization':1,'statistics':1,'python':2,'r':2,'scripting':1,
            'data_warehousing':2,'sql':2,'pig':2,'hive':2,'data_mining':1,
            'tableau':2,'msbi':2,'sas':2,'spss':2,'matlab':2},
            
           {'statistics':1,'linear_algebra':1,'sql':2,'scikit_learn':1,'keras':2,'tensorflow':2,
            'theano':2,'python':1,'r':2,'optimization':2,'deep_learning':2,'nlp':2,'hadoop':2,'hbase':2,'pig':2},
            #hello
           {'statistics':1,'linear_algebra':1,'scikit_learn':1,'keras':1,'tensorflow':1,
            'theano':2,'python':1,'optimization':2,'deep_learning':1,
             'image_processing':1,'digital_signal_processing':1,'opencv':1,'openGL':2,'matlab':1,'c++':1}],
          [[{},{},{}]], #software dev
          [{},{},{}] #game dev
         ]


# In[ ]:


statistics,scikit_learn,linear_algebra,python,excel,data_mining


# In[29]:


#concept dictionary
concept = [
           [{'descriptive_statistics':3,'data_visualization':2,'hypothesis_testing':3,'inferential_statistics':2,
            'optimization':1},
            
           {'linear_regression':1,'logistic_regression':2,'clustering':2,'classification':2,'embedding':1,'regularization':3,
            'feature_engineering':2,'neural_network':3,'data_cleaning':1,'dimensionality_reduction':3,'gradient_descent':2,'bayes_theorem':2},

           {'vector_image':1,'edge_detection':2,'light_color':1,'hough_transform':2,'mathmatical_modeling':3,
            'cnn':2,'deep_networks':2,'dimensionality_reduction':2,'deep_learning':3,'data_cleaning':2}],
          [[{},{},{}]], #software dev
          [{},{},{}] #game dev
         ]


# In[31]:


#cadidate data frame


# In[155]:


import numpy as np
import pandas as pd

interest_domain = ['data science']*3
skill = ['statistics,linear_algebra,sql,scikit_learn,keras,tensorflow',
                         'image_processing,digital_signal_processing,opencv,openGL',
                         'data_visualization,statistics,python,r,scripting,data_warehousing']
concept = ['descriptive_statistics,data_visualization,hypothesis_testing',
           'linear_regression,logistic_regression,clustering,classification,embedding,regularization',
           'vector_image,edge_detection,light_color,hough_transform,mathmatical_modeling']
weight = ['2,1,1,1,1,1',
         '2,2,1,1','2,2,1,2,2,1']

cweight = ['2,2,1','3,3,2,2,2,1','3,3,2,2,3']

d = {'interest' : pd.Series(interest_domain),
     'skill': pd.Series(skill),
     'skill_weights': pd.Series(weight),
         'concept' : pd.Series(concept),
    'concept_weights':pd.Series(cweight)
    }
d = pd.DataFrame(d)


# In[156]:


d


# In[173]:


check = list(skills[0][1].keys())
print(check)


# In[169]:


d.shape


# In[174]:


var = input("Enter your interested Domain")


# In[175]:


var


# In[178]:


suggestion = [[] for i in range(d.shape[0])]
for p in range(d.shape[0]):
    inp = input("Enter your interested domain")
    ## dropdown code skills[ Index for ]
    inp = 1
    if(inp == 'Data Analyst'):
        i = 0
        j = 0
    elif(inp == 'Machine Learning'):
        i = 0
        j = 1
    elif(inp == 'Computer Vision'):
        i = 0
        j = 2
    elif(inp == 'Web Development(Front End)'):
        i = 1
        j = 0
    elif(inp == 'Android Development'):
        i = 1
        j = 1
    elif(inp == 'iOS Development'):
        i = 1
        j = 2
    elif(inp == 'Unity Game Development'):
        i = 2
        j = 0
    elif(inp == 'Android Game Development'):
        i = 2
        j = 1
    elif(inp == 'iOS Game Development'):
        i = 2
        j = 2
    check = list(skills[i][j].keys())  ## this will come candidate's choice  ,i.e. in which domain it wants to go 
    for k in range(len(check)):
        if((check[k] in temp) == False):
            suggestion[p].append(check[k])
    print(suggestion[p])


# In[ ]:


#cv(skill) recommendation
# 1st emplyee wants to go in ML
# 2nd emplyee wants to go in Computer vision
# 3rd emplyee wants to go in DA
