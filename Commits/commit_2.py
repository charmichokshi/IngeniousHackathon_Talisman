
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


# In[4]:


#list of skills
#1: mandatory and 2: supplementary
skill = [[[{'excel':1,'data_visualization':1,'statistics':1,'python':2,'r':2,'scripting':1,
            'data_warehousing':2,'sql':2,'pig':2,'hive':2,'data_mining':1,
            'tableau':2,'msbi':2,'sas':2,'spss':2,'matlab':2},
           {'statistics':1,'linear_algebra':1,'sql':2,'scikit_learn':1,'keras':2,'tensorflow':2,
            'theano':2,'python':1,'r':2,'optimization':2,'deep_learning':2,'nlp':2,'hadoop':2,'hbase':2,'pig':2},
          {}]]


# In[22]:


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


# In[42]:


import numpy as np
a = np.array([[[1, 2,3], [3, 4]], [[5, 6], [7, 8]]])


# In[52]:


import pandas as pd
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = pd.Panel(moveaxis(a,2)).to_frame()
# c = b.set_index(b.index.labels[0]).reset_index()
# c.columns=list('abc')


# In[58]:


df = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),columns=['a', 'b', 'c', 'd', 'e'])


# In[96]:


import numpy as np
import pandas as pd

interest_domain = ['data science']*3


d = {'interest' : pd.Series(interest_domain),
     'skill': pd.Series(['', 2.,2.]),
         'two' : pd.Series([1., 2., 3.])}
d = pd.DataFrame(d)


# In[97]:


d

