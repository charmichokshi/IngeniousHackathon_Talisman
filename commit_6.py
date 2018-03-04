# coding: utf-8

# In[5]:


#data dictionary

#list of fields
field = ['Data Science','Software development','Game dvelopment']


# In[6]:


#list of domains
domain = [['Data Analyst','Machine Learning','Computer Vision'],
          ['Web Development(Front End)', 'Android Development','iOS Development'],
         ['Unity Game Development','Android Game Development', 'iOS Game Development']]


# In[3]:


#ideal skills
skills = [   ### Data Analyst
           [{'excel':1,'data_visualization':1,'statistics':1,'python':2,'r':2,'scripting':1,
            'data_warehousing':2,'sql':2,'pig':2,'hive':2,'data_mining':1,
            'tableau':2,'msbi':2,'sas':2,'spss':2,'matlab':2},
            #### Machine Learning
           {'statistics':1,'linear_algebra':1,'sql':2,'scikit_learn':1,'keras':2,'tensorflow':2,
            'theano':2,'python':1,'r':2,'optimization':2,'deep_learning':2,'nlp':2,'hadoop':2,'hbase':2,'pig':2},
            #### Computer Vision
           {'statistics':1,'linear_algebra':1,'scikit_learn':1,'keras':1,'tensorflow':1,
            'theano':2,'python':1,'optimization':2,'deep_learning':1,
             'image_processing':1,'digital_signal_processing':1,'opencv':1,'openGL':2,'matlab':1,'c++':1}],
            #### Web Development(Front End)
            [{'html':1,'css':1,'javascript':1,'jquery':2,'bootstrap':2,'angular_js':1,'kendo_ui':2,'sinon_js':2,'wordpress':2,
             'mysql':1, 'node.js':1,'mvc':1, 'asp.net':1,'ajax':1, 'jquery':1,'laravel':2,'codeignitor':2 },
            #### Android Development
             {'java':1,'xml':1,'android':1,'android-studio':1,'sql':1,'rest':2, 'soap':2,'json':1,
              'android_sdk':1,'web_services':1},
            #### iOS Development
              {'xcode':1,'swift':1,'objective_c':1,'cocoa_touch':1,'uikit':1,'rest_api':1,'xctest':2,
               'coredata':2,'json':2,'shell_scripting':1,'selenium':2,'ios':1}
            ],
    
          [{},{},{}] #game dev
         ]
         
         


# In[ ]:


skills[s]


# In[87]:


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


# In[92]:


### Candidate Data genration :P
import numpy as np
import pandas as pd

interest_domain = ['data science','data science','data science', 'data science', 'data science',
                   'data science', 'data science', 'data science','Software development','Software development'
                  ,'Software development','Software development','Software development','Software development'
                  ,'Software development']

skill = ['statistics,linear_algebra,sql,scikit_learn,keras,tensorflow', ## ML
                         'image_processing,digital_signal_processing,opencv,openGL', ## ML
                         'data_visualization,statistics,python,r,scripting,data_warehousing', ## Data analytics
        'nlp,python,machine_learning,deep_learning', ## ML
         'data_analysis,excel,mysql,machine_learning,statistics,predictive_modeling,python,r', ## ML
         'aws,programming,java,spark,mysql,hdfs,algorithms,machine_learning,apache,spark,hadoop,javascript,angular_js,html,css',#ML
        'robotics,embedded_c,python,opencv,tensorflow,matlab,object_detection', ##CV
         'c,deep_learning,python,tensorflow,pytorch,c++,mathematical_modelling', ##CV
         'java,c++,c,html,css,javascript,rest_api,json,xml,sqllite,selenium,android_studio,eclipse', ## AD(android development)
         'objective_c,swift,ios,html,css,javascript,xcode,xml,json', ## iOS 
         'swift,ios,python,selenium,shell_scrpting,mysql', ## iOS 
         'c,c++,java,objective_c,swift,xcode,cocoa_touch,xml,json,coredata,sqllite', ## iOS
         'ajax,bootstrap,html,css,angular_js,javascript', ## WD
         'angular_js,sql,dreamweaver,eclipse,html,css,bootstrap,jquery,javascript', ## WD
         'angular_js,html,css,photoshop,ajax,javascript,reactjs,redux' ## WD
        ]
concept = ['descriptive_statistics,data_visualization,hypothesis_testing',
           'linear_regression,logistic_regression,clustering,classification,embedding,regularization',
           'vector_image,edge_detection,light_color,hough_transform,mathmatical_modeling']
weight = ['2,1,1,1,1,1',  ## ML
         '2,2,1,1',  ## Ml
          '2,2,1,2,2,1', ## DA
         '1,2,2,1', ## ML
          '1,2,1,1,1,1,2,1', ## ML
          '1,4,4,2,4,2,4,2,2,2,1,1,1,1,1',  ## ML
          '1,1,1,1,1,1,1' , ## CV
          '3,2,2,2,2,2,2' , ## CV
          '3,3,3,1,1,1,1,1,1,1,1,1,1', ## AD
          '1,1,4,2,2,2,1,1,2', ## iOS
          '1,1,2,1,2,2', ## iOS
          '2,2,3,3,3,3,1,2,1,2', ## iOS
          '2,2,2,2,2,2', ## WD
          '1,2,1,1,2,2,2,1,2', ## WD
          '1,1,1,1,1,1,1,1', ## WD
         ]

cweight = ['2,2,1','3,3,2,2,2,1','3,3,2,2,3']

d = {'interest' : pd.Series(interest_domain),
     'skill': pd.Series(skill),
     'skill_weights': pd.Series(weight),
         'concept' : pd.Series(concept),
    'concept_weights':pd.Series(cweight),
    }
d = pd.DataFrame(d)


# In[93]:


d['domain'] = pd.Series('U', index=d.index)


# In[97]:


d['domainID'] = pd.Series('', index=d.index)


# In[24]:


d.drop(['domain'], axis = 1, inplace = True)


# In[98]:


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


w = widgets.Select(
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


# In[13]:


d.iloc[0][3].split(',')
skills[0]


# In[99]:


#cv(skill) recommendation
# 1st emplyee wants to go in ML
# 2nd emplyee wants to go in Computer vision
# 3rd emplyee wants to go in DA


indexer = ['Data Analyst','Machine Learning','Computer Vision','Web Development(Front End)', 'Android Development','iOS Development']
for i in range(d.shape[0]):
    #print(i)
    score = [0]*6
    skill = d.iloc[i][3].split(',')
    #print(len(skill))
    #print("skill len ",len(d.iloc[i][4].split(',')))
    #print("skill ",skill)
    for j in range(len(skill)):
        flag = 0
        temp = 0
        #print(j)
        for k in range(6):
            #print(k)
            temp = temp + 1
            if(temp==4):
                flag = flag + 1
                temp = 0
            l = list(skills[flag][k%3].keys())
            #print("pair ",flag," ",k%3," ",k)
            if((skill[j] in l) == True):
                score[k] = score[k] + int(d.iloc[i][4].split(',')[j])
    print(score)
    d.loc[i]['domain'] = indexer[np.argmax(score)]
    d.loc[i]['domainID'] = np.argmax(score)


# In[7]:


indexer = ['Data Analyst','Machine Learning','Computer Vision','Web Development(Front End)', 'Android Development','iOS Development']
indI = [0, 0, 0, 1,1, 1]
indJ = [0, 1, 2,0, 1, 2]


# In[111]:


for i in range(d.shape[0]):
    skill = d.iloc[i][3].split(',')
    domainID = d.loc[i]['domainID']
    for j in range(len(skill)):
        if((skill[j] in skills[ indI[domainID] ] [ indJ[domainID] ]) == False ):
            skills[ indI[domainID] ] [ indJ[domainID] ].update({skill[j]: 2})
            


# In[116]:


skills[1]


# In[121]:


## deleting web development entries
del skills[1][0]['hadoop']s
del skills[1][0]['hdfs']
del skills[1][0]['machine_learning']
del skills[1][0]['spark']
del skills[1][1]['css']
del skills[1][1]['rest']
del skills[1][2]['cocoa-touch']
del skills[1][2]['core-data']
del skills[1][2]['objective-c']
del skills[1][2]['python']

### We successfuully deleted the skills entries which shouldn't be in the IDEAL SKILL SET


# In[124]:


d['skill']


# In[110]:


tempp = {'hey':2, 'bey':3}
tempp.update({'ch':8})


# In[4]:


#skill recommendation (IMPROVED)

test = ['python','linear_algebra','scikit_learn','keras','tensorflow',
        'image_processing','mathematical_modelling','opencv','opengl']
test_weight = [2,1,1,1,2,2,3,3,2]

test_score = [0]*6

for i in range(len(test)):
    flag = 0
    temp = 0    
    for k in range(6):
                #print(k)
                temp = temp + 1
                if(temp==4):
                    flag = flag + 1
                    temp = 0
                l = list(skills[flag][k%3].keys())
                #print("pair ",flag," ",k%3," ",k)
                if((test[i] in l) == True):
                    test_score[k] = test_score[k] + test_weight[i]


# In[9]:


import numpy as np
test_domain = indexer[np.argmax(test_score)]
test_domain


# In[ ]:


#employee recommendation based on score
#get job description
#give score to each candidate for that job description
#pick up candidate with high score

#here give score based on importance of skillset and concept set
#importance is derived from giving weight to skills
#high weight skills are taken from important employee skill set and ideal skill set


# In[ ]:


#employee recommendation based on similarity
#get candidate skill set and concept set
#get skill set of employees of similar type of company
#rate those employees
#pick top employees and find similar employees in our database
#pick best suitable candidate for them (using employee skill data)


# In[ ]:


#company recommendation
#get candidate skill set
#find nearest employee set and pick most common company occurence and give it to the user


# In[ ]:


#Domain(Job) recommendation (done)
#get candidate skill set
#give different score to this candidate in different domain
#pick highest score domain and say user to do job in this domain

#skill(Job) recommendation (done)
#1. give missing skills from ideal skill set
#2. use candidate data and give them domain labels using "Domain Recommendation algo"
#   now update ideal skill set to capture trending skill set 
#   after this use updated ideal skill set to recommend missings skills (IMPROVED version)
