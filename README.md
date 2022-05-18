# Analyze-Data-TedTalk-in-Kaggle-Website
These datasets contain information about all audio-video recordings of TED Talks uploaded to the official TED.com website until September 21st, 2017. The TED main dataset contains information about all talks including number of views, number of comments, descriptions, speakers and titles. The TED transcripts dataset contains the transcripts for all talks available on TED.com

There are two CSV files.

ted_main.csv - Contains data on actual TED Talk metadata and TED Talk speakers.

transcripts.csv - Contains transcript and URL information for TED Talks

### 1-COLECT DATA
##### Create the connection between google colab and google drive
from google.colab import drive
drive.mount('/content/drive')

##### Attach the library
import pandas as pd
import numpy as np
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
import re

##### Create the linking tables
main_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ted_main.csv')
trans_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/transcripts.csv')

##### Merge the ted_main and transcripts 
df = pd.merge(main_df, trans_df, how = 'left', on = 'url')

### 2-CLEAN DATA 
##### Dropping unnecessary columns including url and description in a transcripts file
##### Changing the original DataFrame, use the inplace = True argument
df.drop(columns=['url'],inplace=True)

##### Arrange order of columns
df = df[['name', 'title', 'main_speaker', 'num_speaker', 'speaker_occupation', 'duration', 'film_date', 'published_date', 'event', 'comments', 'tags', 'languages', 'ratings', 'views', 'transcript', 'description']]

##### Change numerical format
pd.options.display.float_format = "{:.2f}".format

##### Change the time of duration to minutes   
df['duration'] = round(df['duration']/60, 2)

##### Change the time of film_date, and published_date from senconds to yyyy-mm-dd and hh:mm:ss
df['film_date'] = pd.to_datetime(df['film_date'], unit = 's')                  # df.film_date 
df['published_date'] = pd.to_datetime(df['published_date'], unit = 's')        # df.published_date

#####Separate day, month, year of the columns of film_date & published_date
df['Year'] = df['published_date'].dt.year
df['Year_Month_filming']=df['film_date'].dt.to_period('M')
df['Year_Month_publishing']=df['published_date'].dt.to_period('M')

##### Replace missing data by scalar value 
df['speaker_occupation'] = df['speaker_occupation'].fillna('Unknown')
df['transcript'] = df['transcript'].fillna('Unknown')

df.isnull().sum()

![image](https://user-images.githubusercontent.com/103476246/168977426-444d27d9-5dc6-4232-b727-5f29cbb3349f.png)

df.info()

![image](https://user-images.githubusercontent.com/103476246/168978087-f58aa400-b241-4585-8db0-597c854ee4d9.png)

df.describe()

![image](https://user-images.githubusercontent.com/103476246/168978177-a1641211-099a-4d93-a60f-2edb78dc9851.png)


###### Step 1: Normalization
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
###### Step 2: Stemming or Lemmatizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer() #Lemmatizer
###### Step 3: Remove stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = stopwords.words("english")

##### Clean title, transcript, description
def clean_text(x):
###### Step 1: Normalization
###### Remove HTML 
    raw = BeautifulSoup(x, "html.parser")
    raw = raw.get_text()
###### Remove non-alphabetic & lower
    clean = re.sub(r"(Laughter)"," ", raw)
    clean = re.sub("[^a-zA-Z ]", " ", clean).lower()
###### Step 2-3: Lemmatize and remove stopwords
    clean = [lemmatizer.lemmatize(word) for word in clean.split() if word not in stops]
    return ' '.join(clean)
for text in ['title', 'transcript', 'description']:
    df[text] = df[text].apply(clean_text)

##### Convert the column of rating as a JSON object into the column of each rating row 
##### Get the 1st representating row 
##### Raplace ' into " to transfer numerical values into string values,  i.g. 'count': 19645
df['ratings']=df['ratings'].str.replace("'",'"')       
##### Use JSON to extract data
pd.read_json(df['ratings'][0])[['id','name','count']]

![image](https://user-images.githubusercontent.com/103476246/168979211-19cec6f9-4c6c-4212-8e0b-c0c027f01063.png)

### 3-EXPLORATORY DATA ANALYSIS

##### The correlation of comments and views
df[['comments','views']].corr()

![image](https://user-images.githubusercontent.com/103476246/168979511-05795d96-96b8-4bfd-9917-1b8c518e7c51.png)

##### The correlation between Comments and Views
plt.figure(figsize = (10,5))
plt.title('The correlation between Comments and Views')
sns.scatterplot(data = df,
                x = 'views',
                y = 'comments'
               )
plt.xlabel('Number of views')
plt.ylabel('Number of comments')
plt.savefig('The correlation between Comments and Views.png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168979696-ff506659-2aae-4bf5-b696-f2ba53b74770.png)

##### The 5 most popular topical issues in Ted Talk 
dummies = df['tags'].str.get_dummies(sep = ',')
plt.figure(figsize = (10,5))
plt.title('The 5 most popular topical issues in Ted Talk ')
sns.barplot(data = dummies.sum().reset_index().sort_values(0, ascending = False).head(),
            x = 0,
            y = 'index')
plt.xlabel('Frequency of Appearance')
plt.ylabel('Topical Issues')
plt.savefig('The 5 most popular topical issues in Ted Talk.png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168979884-4f1b79a1-cd74-4ba4-bc1d-41fe89cbcd25.png)

##### There are 5 most common occupations in TED TALK show
df['speaker_occupation'].value_counts().head(10).reset_index()

![image](https://user-images.githubusercontent.com/103476246/168980021-b9d49763-eb7a-4ec9-9a06-7d397a45021c.png)

##### Appearance of speaker occupation in TEDTALK 
dummies = df['speaker_occupation'].str.get_dummies()
data = dummies.sum().reset_index().sort_values(0, ascending = False).head(10)
data

![image](https://user-images.githubusercontent.com/103476246/168980257-98586be9-163a-401b-b5ca-d6208b7243f2.png)

##### The frequency of appearance of speaker occupation in Ted Talk
dummies = df['speaker_occupation'].str.get_dummies()
plt.figure(figsize = (10,10))
plt.suptitle('The facts of the most popular speaker occupations and \n speaker occupations has highest views')
plt.subplot(211)
plt.title('The 5 most popular speaker occupations in Ted Talk ')
sns.barplot(data = dummies.sum().reset_index().sort_values(0, ascending = False).head(),
            x = 0,
            y = 'index')
plt.xlabel('Frequency of Appearance')
plt.ylabel('Speaker Occupations')
print()
##### The 5 speaker occupations has most views
plt.subplot(212)
plt.title('The 5 speaker occupations has highest views in Ted Talk ')
sns.barplot(x = 'views',
            y = 'speaker_occupation', 
            data = df.sort_values('views', ascending = False)[:5])
plt.xlabel('Number of Views')
plt.ylabel('Speaker Occupations')
plt.savefig('The fact of the most popular speaker occupations and speaker occupations has highest views .png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168980462-7e4d8ff2-9556-41c1-be28-66126231e382.png)

data = df.sort_values('views', ascending = False)[['speaker_occupation','views']][:5]
data

![image](https://user-images.githubusercontent.com/103476246/168980559-8ab15f42-2b28-47b4-8188-4b493f233189.png)

data = df.groupby('num_speaker')['views'].mean().reset_index()
data

![image](https://user-images.githubusercontent.com/103476246/168980803-ab2ea20f-b7e1-4d27-912b-de3afaa3a231.png)

##### The affection of number of speakers and views  
##### The lower the number of speakers in the video, the better efficency
plt.figure(figsize =(15,6))
plt.suptitle('The Impact of number of speakers to views')
plt.subplot(121)
plt.title('Views are evaluated by number of speakers')
sns.barplot( data = df.groupby('num_speaker')['views'].mean().reset_index(),
            x = 'num_speaker',
            y = 'views')
plt.xlabel('Number of Speakers')
plt.ylabel('Number of Views')

plt.subplot(122)
plt.title('Performance is determined by dispersion of elements')
sns.scatterplot(data = df,
            x = 'num_speaker',
            y = 'views')
plt.xlabel('Number of Speakers')
plt.ylabel('Number of Views')
plt.savefig('The Impact of number of speakers to views.png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168980892-34ca221d-2798-4917-975d-0e519343920a.png)

##### Top 5 main speakers which has views most 
##### Main_speaker vs Views
df.sort_values('views', ascending = False)[['main_speaker','views']].head()

##### Case 1: Consider relationship between main speaker and speaker occupation with 5 videos having most popular views in Ted Talk
###### Main_speaker - Views
plt.figure(figsize=(15, 10))
plt.suptitle('Relationship between Main Speaker and Speaker Occupation \n Five videos have the most popular views in Ted Talk')
plt.subplot(211)
plt.title('Relationship between main speaker and view')
sns.barplot( data = df.sort_values('views', ascending = False).head(5),
             x = 'views',
             y = 'main_speaker')
plt.xlabel('Number of views')
plt.ylabel('Main Speakers')

###### Occupation - Views
plt.subplot(212)
plt.title('Relationship between speaker occupation and view')
sns.barplot(x = 'views',
            y = 'speaker_occupation', 
            data = df.sort_values('views', ascending = False)[:5])
plt.xlabel('Number of views')
plt.ylabel('Speakers Occupations')
plt.savefig('Relationship between Main Speaker and Speaker Occupation which has most popular video view.png', transparent=False, dpi=80, bbox_inches="tight")

##### Case 2: Consider relationship between main speaker and speaker occupation with 5 videos having least popular views in Ted Talk
###### Main_speaker - Views
plt.figure(figsize=(15, 10))
plt.suptitle('Relationship between Main Speaker and Speaker Occupation \n Five videos have the least views in Ted Talk')
plt.subplot(211)
plt.title('Relationship between main speaker and view')
sns.barplot( data = df.sort_values('views', ascending = True).head(5),
             x = 'views',
             y = 'main_speaker')
plt.xlabel('Number of views')
plt.ylabel('Main Speakers')

###### Occupation - Views
plt.subplot(212)
plt.title('Relationship between speaker occupation and view')
sns.barplot(x = 'views',
            y = 'speaker_occupation', 
            data = df.sort_values('views', ascending = True)[:5])
plt.xlabel('Number of views')
plt.ylabel('Speakers Occupations')
plt.savefig('Relationship between Main Speaker and Speaker Occupation which has least video view.png', transparent=False, dpi=80, bbox_inches="tight")


##### Duration a lot, views has not sure a lot. From here we will visualize the factors below following 4 cases
df.sort_values('views', ascending = False)[['main_speaker','speaker_occupation','title','languages','duration','comments','views']].head(5)

![image](https://user-images.githubusercontent.com/103476246/168981440-9d78b6af-db5d-4776-ae08-d876ad5f1570.png)

##### 5 speaker_occupations have the lowest view
df.sort_values('views', ascending = True)[['main_speaker','speaker_occupation','title','languages','duration','comments','views']].head(5)

![image](https://user-images.githubusercontent.com/103476246/168981558-1ba54e5a-e902-4dfa-96c2-efa09f200b84.png)


##### Relationship between number of languages and views
plt.figure(figsize = (10,5))
plt.suptitle('Relationship between number of languages and views \n')
plt.subplot(121)
plt.title('Bar Chart')
sns.barplot(x="languages", y="views", data = df.sort_values(['languages','views'], ascending=False)[:10])
plt.xlabel('Number of languages')
plt.ylabel('Number of views')

plt.subplot(122)
plt.title('Line Chart')
sns.lineplot(x = 'languages', y='views', data = df.groupby('languages')['views'].mean().reset_index())
plt.xlabel('Number of languages')
plt.ylabel('Number of views')
plt.savefig('Relationship between languages and views.png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168981798-b42e54b2-2efb-4960-9be4-49dbd3a12f90.png)

##### The views achieve maximum when video has 60 languages 
##### There are many languages, it has not sure many views
Plot_1_1 = sns.lineplot(x = 'languages', y='views', data = df.groupby('languages')['views'].mean().reset_index())

![image](https://user-images.githubusercontent.com/103476246/168981890-5f618a33-8171-4fb8-a504-24c25f32b3ab.png)


##### Let's see which video got the most views via main speakers
Plot_2 = sns.barplot(x="views", y="main_speaker", data = df.sort_values('views', ascending=False)[:20])

![image](https://user-images.githubusercontent.com/103476246/168982072-5a1882c6-b413-4b46-a7a0-fd576698a8fd.png)

##### Total Video Appears via Years
video_of_year = df.groupby('Year').count()['name'].reset_index()
video_of_year.rename(columns = {'name': 'number of videos'}, inplace = True)
video_of_year

![image](https://user-images.githubusercontent.com/103476246/168982264-b22da8f2-b08a-4309-ae8f-48ce5f593f74.png)

##### VISUALIZE THE NUMBER OF VIDEOS VIA YEARS
plt.figure(figsize =  (15, 8))
sns.barplot(data = video_of_year,
            x = 'Year',
            y = 'number of videos'
           )
           
![image](https://user-images.githubusercontent.com/103476246/168982493-251eee4b-4a34-44d7-b624-9bafeac27654.png)

##### Total Number of Views via Years
df.groupby('Year')['views'].sum().reset_index()

![image](https://user-images.githubusercontent.com/103476246/168983026-60d2c578-ef48-45f1-8f30-ba28d2e0727d.png)

##### VISUALIZE THE TOTAL NUMBER OF VIEWS VIA YEARS  

plt.figure(figsize =  (15, 8))
sns.barplot(data = df.groupby('Year')['views'].sum().reset_index(),
            x = 'Year',
            y = 'views')
            
![image](https://user-images.githubusercontent.com/103476246/168983262-95534b0f-3c45-4795-8520-9be639dd1d2a.png)

# VISUALIZE THE NUMBER OF VIDEOS VIA YEARS
plt.figure(figsize =  (15, 6))
plt.suptitle('NUMBER OF VIDEOS and VIEWS via YEARS')
sns.lineplot(data = video_of_year,
             x = 'Year', 
             y = 'number of videos',
             color ='red'
             )#.legend(['Video'])
plt.xticks(ticks = range(2006,2018,1))
plt.ylabel('Number of videos')

#VISUALIZE THE TOTAL NUMBER OF VIEWS VIA YEARS  
plt.twinx() 
sns.lineplot(data = df.groupby('Year')['views'].sum().reset_index(),
             x = 'Year',
             y = 'views',
             color ='blue'
             )
plt.ylabel('Number of views')
plt.xlim([2005, 2018])
plt.style.use('default')
plt.savefig('Number of videos and views via Year.png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168983445-229dc203-aecb-473f-924e-62040e76954c.png)

#####  Months — Talk show hosted the most

talk_month = pd.DataFrame(df['film_date'].map(lambda x: x.month).value_counts()).reset_index()
talk_month.columns = ['Month', 'Talk Shows']
talk_month

![image](https://user-images.githubusercontent.com/103476246/168983890-fcc91e21-04f4-441b-a656-46c3f5f6f71f.png)

##### Number of videos via months from 2006 to 2017
plt.figure(figsize = (15, 8))
plt.title('Number of videos via months from 2006 to 2017')
sns.barplot(x='Month', y='Talk Shows', data = talk_month)
plt.ylabel('Number of videos')
plt.savefig('Number of videos and views via months from 2006 to 2017.png', transparent=False, dpi=80, bbox_inches="tight")

![image](https://user-images.githubusercontent.com/103476246/168984080-80adc832-c9d7-4c41-b8ef-1deb9e46fb28.png)





            

           

