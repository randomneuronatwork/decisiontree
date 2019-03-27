#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[ ]:


df=pd.read_csv("C:/Users/steve/Desktop/hr data/b.csv")
sc=StandardScaler()



# In[ ]:


y=df.iloc[:,[0]]
x=df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]
sc.fit_transform(x)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=3)
clf=clf.fit(x_train,y_train)
#from xgboost import XGBClassifier
#clf=XGBClassifier()
#clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
print(accuracy_score(y_test,y_predict))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
print('confusion matrix-\n')
print(cm)



# In[ ]:



dot_data=StringIO()
feature=['AttritionAge','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'
]
clas=['yes','no']
tree.export_graphviz(clf,out_file=dot_data,feature_names=feature,class_names=clas,filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
g=Image(graph.create_png())
display(g)

