#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
for dirname, _, filenames in os.walk('C:/Users/gobin/Downloads/MAchine learning Project - STAT 4650'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[65]:


data = pd.read_csv('C:/Users/gobin/Downloads/MAchine learning Project - STAT 4650/AB_US_2023.csv')

data


# In[ ]:





# In[66]:


import folium
from sklearn.cluster import KMeans

MapModel = data[['latitude', 'longitude']]

# Instantiate the clustering model and visualizer
model = KMeans()

kmeans = KMeans(n_clusters = 15, random_state=42).fit(MapModel)
kmeans.cluster_centers_

cluster_map = folium.Map([41.8781, -87.6298], zoom_start=4)
for i in range(kmeans.cluster_centers_.shape[0]):
    num = sum(kmeans.labels_ == i)
    folium.CircleMarker([kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1]],
                        radius=15,
                        popup=str(num) + ' Listings Associated with this Cluster',
                        fill_color="#3db7e4", # divvy color
                        ).add_to(cluster_map)
cluster_map


# In[67]:


rows=2
cols=5
fig,ax=plt.subplots(rows,cols,figsize=(20,8))
d=data.select_dtypes(include=['float64','int64']).columns
for row in range(rows):
    for col in range(cols):
        ax[row,col].hist(data[d[cols*row+col]])
        ax[row,col].set_xlabel(d[cols*row+col])
plt.show()


# In[68]:


import matplotlib.pyplot as plt
import numpy as np

x = data['price']

plt.hist(x, color = '#8B0000', edgecolor= 'black')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show() 


# In[69]:


sns.displot(data[(data['price']<200) & (data['price']>40)]['price'],kde=False,bins=10, color = '#8B0000')


# In[ ]:





# In[70]:


rows=2
cols=5
fig,ax=plt.subplots(rows,cols,figsize=(20,8))
d=data.select_dtypes(include=['float64','int64']).columns
for row in range(rows):
    for col in range(cols):
        ax[row,col].hist(data[d[cols*row+col]])
        ax[row,col].set_xlabel(d[cols*row+col])
plt.show()


# In[71]:


city=data.groupby('city').agg({'id':'count','price':np.mean,'number_of_reviews':['sum',np.mean],'availability_365':np.mean})
city


# In[72]:


fig=plt.figure(figsize=(10,10))
ax=sns.barplot(y=city.index,x=city.id['count'], color = 'Blue')
ax.set_xlabel('Listing from respective cities')
ax.set_title('Number of Listing vs cities')
plt.show()


# In[73]:


fig=plt.figure(figsize=(10,10))
ax=sns.barplot(y=city.index,x=city.price['mean'])
ax.set_xlabel('Average price per night from respective cities')
ax.set_title('Price vs city')
plt.show()


# In[74]:


fig=plt.figure(figsize=(10,10))
ax=sns.barplot(y=city.index,x=city.availability_365['mean'])
ax.set_xlabel('Average availablitiy over the year from respective cities')
ax.set_title('Availability vs city')
plt.show()


# In[75]:


fig=plt.figure(figsize=(10,10))
ax=sns.barplot(y=city.index,x=city.number_of_reviews['mean'], color = 'Orange')
ax.set_xlabel('Average number of reviews of listings in city')
ax.set_title('Average number of reviews vs city')
plt.show()


# In[76]:


states_dic = {'Asheville':'NC','Austin':'TX','Boston':'MA','Broward County':'FL','Cambridge':'MA','Chicago':'IL','Clark County':'NV','Columbus':'OH','Denver':'CO','Hawaii':'HI','Jersey City':'NJ',
             'Los Angeles':'SC','Nashville':'TN','New Orleans':'MS','New York City':'NY','Oakland':'CA','Pacific Grove':'CA','Portland':'OR','Rhode Island':'RI','Salem':'MA','Santa Clara County':'CA',
             'Santa Cruz County':'CA','San Diego':'CA','San Francisco':'CA','San Mateo County':'CA','Seattle':'WA','Twin Cities MSA':'MN','Washington D.C.':'DC'}
data['States']=data['city'].apply(lambda x: states_dic[x])


# In[77]:


pip install plotly


# In[78]:


states=data.groupby("States").agg({'id':'count','price':np.mean,'number_of_reviews':['sum',np.mean],'availability_365':np.mean})


# In[79]:


import plotly.express as ex
fig=ex.choropleth(locations=states.index,color=states.price['mean'],locationmode='USA-states',title="Average Airbnb price of listings from states",scope='usa',color_continuous_scale=ex.colors.diverging.Portland)
fig.show()


# In[80]:


from tensorflow.keras.preprocessing.text import one_hot,text_to_word_sequence,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,LSTM,Dropout,Bidirectional
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow.keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestRegressor
import nltk
import re 


# In[81]:


pip install tensorflow


# In[82]:


from tensorflow.keras.preprocessing.text import one_hot,text_to_word_sequence,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Embedding,LSTM,Dropout,Bidirectional
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow.keras.optimizers import RMSprop
from sklearn.ensemble import RandomForestRegressor
import nltk
import re


# In[83]:


data['nlp_text']=data['name']+" "+data['room_type']
data['nlp_text']=data['nlp_text'].astype('str').apply(lambda x:x.lower())  #converting to lower case
data['nlp_text']=[re.sub('[^A-Za-z]'," ",x) for x in data['nlp_text']] #removing characters other than alphabets


# In[84]:


nltk.download('stopwords')
import nltk


# In[85]:


def remove_stopwords(x):
    h=[]
    x=x.split(" ")
    STOP=stopwords.words("english")
    h=[j for j in x if j not in STOP]
    return " ".join(h)
data['nlp_text']=data['nlp_text'].apply(lambda x:remove_stopwords(x)) 


# In[86]:


s=set()
for x in data['nlp_text']:
    words=x.split(" ")
    for y in words:
        s.add(y)
print("Total number of unique words :",len(s)) 


# In[87]:


lengths=[len(x.split(" ")) for x in data['nlp_text']] 
sns.distplot(lengths,kde=False,bins=10)


# In[88]:


np.mean(lengths)


# In[89]:


tokenizer=Tokenizer(num_words=25000)
tokenizer.fit_on_texts(s)
sequences=tokenizer.texts_to_sequences(data['nlp_text'])
input_len=10 #keeping only the first 10 words 
sequences=pad_sequences(sequences,input_len,padding='post',truncating='post')


# In[90]:


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=25000)
tokenizer.fit_on_texts(s)
sequences = tokenizer.texts_to_sequences(data['nlp_text'])
input_len = 10  # keeping only the first 10 words 
sequences = pad_sequences(sequences, input_len, padding='post', truncating='post')

model = Sequential()
model.add(Embedding(25000, 64))  # Specify input_length here
model.add(Flatten())
model.add(Dense(480, activation='relu'))
model.add(Dense(1))


# In[91]:


test=sequences[120000:]
train=sequences[:120000]
y_train=data['price'][:120000]
y_test=data['price'][120000:]


# In[92]:


model.compile(metrics=['mae'], optimizer='adam', loss='mse')
model.fit(train, y_train, epochs=5, batch_size=150, validation_split=0.3)


# In[93]:


test_set=data.iloc[120000:]
test_set['predicted']=model.predict(test)
mean_absolute_error(test_set['price'],test_set['predicted'])


# In[94]:


df2=data[['minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','room_type','city']]


# In[95]:


dummy=pd.get_dummies(df2[['room_type','city']])
df2=pd.merge(df2,dummy,right_index=True,left_index=True)


# In[96]:


y=data['price']
df2.replace({np.nan:0},inplace=True)


# In[97]:


X_train=df2.iloc[:120000]
X_test=df2.iloc[120000:]
y_train=data['price'][:120000]
y_test=data['price'][120000:]


# In[98]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Assuming 'data' is your original DataFrame

# Select columns for features and target
df2 = data[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'room_type', 'city']]
y = data['price']

# One-hot encode categorical variables
dummy = pd.get_dummies(df2[['room_type', 'city']])
df2 = pd.concat([df2, dummy], axis=1)
df2.drop(['room_type', 'city'], axis=1, inplace=True)

# Replace NaN values with 0
df2.fillna(0, inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.2, random_state=42)

# Fit the RandomForestRegressor
forest = RandomForestRegressor(max_depth=10, n_estimators=250)
forest.fit(X_train, y_train)


# In[99]:


test_set.reset_index(drop=True, inplace=True)
test_set['forest_predicted'] = forest.predict(X_test[:len(test_set)])


# In[ ]:


from sklearn.metrics import mean_absolute_error

# Assuming you have already reset the index of test_set
# and y_test is the correct subset of target values for the test set
mae = mean_absolute_error(test_set['forest_predicted'], y_test[:len(test_set)])
print("Mean Absolute Error:", mae)


# In[ ]:


test_set['average_prediction']=0.5*test_set['forest_predicted']+0.5*test_set['predicted']
mean_absolute_error(test_set['average_prediction'],test_set['price'])


# In[ ]:


test_set.iloc[:15][['price','predicted','forest_predicted','average_prediction']]


# In[ ]:


mean_squared_error(test_set['average_prediction'],test_set['price'])**0.5


# In[ ]:


test_set


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(data['nlp_text'])


X_train, X_test, y_train, y_test = train_test_split(X, data['price'], test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


feature_importance = model.coef_


feature_names = tfidf.get_feature_names_out()
feature_importance_mapping = dict(zip(feature_names, feature_importance))

sorted_features = sorted(feature_importance_mapping.items(), key=lambda x: x[1], reverse=True)

top_n = 10
top_features = dict(sorted_features[:top_n])
plt.figure(figsize=(12, 6))
plt.bar(top_features.keys(), top_features.values(), color = 'grey')
plt.xticks(rotation=45)
plt.xlabel('Amenity')
plt.ylabel('Impact on Price')
plt.title(f'Top {top_n} Amenities Impact on Price')
plt.show()


# In[103]:


pip install wordcloud


# In[106]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming 'data' is your pandas DataFrame
text = ' '.join(data['nlp_text'].astype(str))

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

# Display the word cloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




