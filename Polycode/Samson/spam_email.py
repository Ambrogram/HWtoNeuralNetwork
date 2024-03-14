import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('D:/Study Files/NEU\Academic/INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]/spam.csv', encoding='latin-1')

print(df.shape)

df.head()

df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

wordcloud = WordCloud(background_color='white', width=800, height=400).generate(''.join(df.v2))

plt.figure(figsize=(20, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

sns.countplot(x = df["v1"], data = df)

df["v1"].value_counts()

# It seems that the given dataset is imbalanced.

4825 // 747

df['v1'] = df["v1"].map({'spam':1,'ham':0})

from sklearn.utils import resample

# create two different dataframe of majority and minority class 

df_majority = df[(df['v1'] == 0)] 

df_minority = df[(df['v1'] == 1)] 

# upsample minority class
df_minority_upsampled = resample(df_minority,
                                 
replace = True,    # sample with replacement  
                                 
 n_samples = 4825, # to match majority class     
                                 
 random_state = 42) 

# reproducible results
    
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled.isnull().sum()


# Let's try to convert these text into vectors using bag of words

text = ['Hello my name is james', 'james this is my python notebook', 'james trying to create a big dataset', 'james of words to try differnt', 'features of count vectorizer']

vectorizer = CountVectorizer(stop_words='english')

count_matrix = vectorizer.fit_transform(text)

count_array = count_matrix.toarray()

df1 = pd.DataFrame(data = count_array,columns = vectorizer.get_feature_names())

df1

# Remove the stop words and transform the texts into the vectorized input variables X

X = vectorizer.fit_transform(df["v2"])

y = df["v1"]

# Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.3, random_state=0)


clf = GaussianNB()

clf.fit(X_train, y_train)

clf.score(X_test, y_test)