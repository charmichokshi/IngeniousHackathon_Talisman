
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


# In[3]:


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
          [[{'html':1,'css':1,'javascript':1,'jquery':2,'bootstrap':2,'angular-js':1,'kendo-ui':2,'sinon-js':2},
            {'java':1,'xml':1,'android':1,'android-studio':1,'sql':1,''},{}]], #software dev
          [{},{},{}] #game dev
         ]


# In[ ]:


import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)


# In[4]:


statistics,scikit_learn,linear_algebra,python,excel,data_mining


# In[5]:


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


# In[6]:


### Candidate Data genration :P
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


# In[7]:


d


# In[8]:


check = list(skills[0][1].keys())
print(check)


# In[42]:


temp = d.iloc[0][3].split(',')
temp


# In[9]:


d.shape


# In[10]:


import ipywidgets as widgets
from IPython.display import display


# In[62]:


Id = input("Enter Candidate ID")


# In[63]:


w=widgets.Select(
        options=['Data Analyst','Machine Learning','Computer Vision',
                     'Web Development(Front End)','Android Development','iOS Development',
                     'Unity Game Development','Android Game Development','iOS Game Development'],
        value='Machine Learning',
        # rows=10,
        description='Domain Choice:',
        disabled=False
    )
display(w)
# w.value
# print("INPPP", inp)




# In[66]:


inp = w.value


# In[67]:


suggestionSkills = []
# suggestionConcepts = []
p = int(Id)
#     inp = input("Enter your interested domain")
## dropdown code skills[ Index for ]
#     inp = 1
candidateSkills = d.iloc[p][3].split(',')
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
checkSkills = list(skills[i][j].keys())  ## this will come candidate's choice  ,i.e. in which domain it wants to go 
# checkConcepts = list(skills[i][j].keys())
for k in range(len(checkSkills)):
    if((checkSkills[k] in candidateSkills) == False):
        suggestionSkills.append(checkSkills[k])
print("Suggested skills for candidate",Id, "in",inp, "are")
for j in range(len(suggestionSkills)):
    print(j+1 ," ",suggestionSkills[j])


# In[ ]:


#cv(skill) recommendation
# 1st emplyee wants to go in ML
# 2nd emplyee wants to go in Computer vision
# 3rd emplyee wants to go in DA



## To wait after user input  
import asyncio
from ipykernel.eventloops import register_integration

@register_integration('asyncio')
def loop_asyncio(kernel):
    '''Start a kernel with asyncio event loop support.'''
    loop = asyncio.get_event_loop()

    def kernel_handler():
        loop.call_soon(kernel.do_one_iteration)
        loop.call_later(kernel._poll_interval, kernel_handler)

    loop.call_soon(kernel_handler)
    try:
        if not loop.is_running():
            loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        
%gui asyncio 

import asyncio
def wait_for_change(widget, value):
    future = asyncio.Future()
    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)
    widget.observe(getvalue, value)
    return future

w=widgets.Select(
        options=['Data Analyst','Machine Learning','Computer Vision',
                     'Web Development(Front End)','Android Development','iOS Development',
                     'Unity Game Development','Android Game Development','iOS Game Development'],
        value='Machine Learning',
        # rows=10,
        description='Domain Choice:',
        disabled=False
    )

async def f():
    for i in range(10):
        #print('did work %s'%i)
        x = await wait_for_change(w, 'value')
        print(i," ",x)
asyncio.ensure_future(f())

w
