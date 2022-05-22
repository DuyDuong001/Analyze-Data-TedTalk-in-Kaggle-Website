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

### EXPLORATORY DATA ANALYSIS
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

##### VISUALIZE THE NUMBER OF VIDEOS VIA YEARS
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

##### Convert date
weekday ={0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thur', 5: 'Fri', 6: 'Sat'}
def convert_date(date):
    return datetime.utcfromtimestamp(date).strftime('%Y-%m-%d')    
    
df['published_date'] = pd.to_datetime(df['published_date'].apply(convert_date))
df['film_date'] = pd.to_datetime(df['film_date'].apply(convert_date))
df['year_published'] = df['published_date'].dt.year
df['published_month']= df['published_date'].dt.month
df['film_month'] = df['film_date'].dt.month
df['day_range'] = (df['published_date'] - df['film_date']).dt.days
df['published_weekday'] = df['published_date'].dt.dayofweek 
df['published_weekday'] = df['published_weekday'].apply(lambda x: weekday[x])

##### Convert ratings -> dict
def convert_ratings(ratings):
    return dict(re.findall("'name': '(\D+)', 'count': ([0-9]+)", ratings))
df['ratings'] = df['ratings'].apply(convert_ratings)

##### Rating filter: positive - negative ratings
rating_name = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded',
       'Confusing', 'Informative', 'Fascinating', 'Unconvincing',
       'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring']
for i in rating_name:
    df[i] = df['ratings'].apply(lambda ratings: int(ratings[i]))
positive_rating = ['Funny','Beautiful','Ingenious','Courageous','Informative','Fascinating','Persuasive','OK','Inspiring']
negative_rating = ['Longwinded','Confusing','Unconvincing','Obnoxious'] # Jaw-dropping
sum_positive = 0
sum_negative = 0

for positive in positive_rating:
    sum_positive += df[positive]
    df['positive_rating'] = sum_positive
for negative in negative_rating:
    sum_negative += df[negative]
    df['negative_rating'] = sum_negative
df.drop(columns = rating_name, inplace = True)

##### Divide into 2 groups: million-view video and under-million-view
def divide(view):
    if view >= 10**6:
        return 'million-view'
    else:
        return 'under-million-view'
df['type'] = df['views'].apply(divide)

##### Engage Rate
df['engage_rate']= df['comments'] / df['views'] * 100.0            

##### Negative/Positive Ratio
df['negative/positive ratio'] = df['negative_rating'] / df['positive_rating']
           
##### Positive/Negative Ratio
df['positive/negative ratio'] = df['positive_rating'] / df['negative_rating']

plt.figure(figsize=(20,6))
sns.heatmap(data= df.corr(), annot=True, fmt='.2f')

![image](https://user-images.githubusercontent.com/103476246/169678156-edde3538-21eb-4603-834b-4551fe9ad4c9.png)

##### Top những rating phổ biến => Video tedtalk đa số sẽ truyền cảm hứng cho audience
rating_list = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded',
       'Confusing', 'Informative', 'Fascinating', 'Unconvincing',
       'Persuasive', 'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring']
rating_rank = {}
for i in df['ratings']:
    for j in rating_list:
        if j not in rating_rank.keys():
            rating_rank[j] = int(i[j])
        if j in rating_rank.keys():
            rating_rank[j] += int(i[j])

plt.figure(figsize = (20,6))
sns.barplot(data =pd.Series(rating_rank).reset_index(), x= 'index', y= 0)

![image](https://user-images.githubusercontent.com/103476246/169678198-abe69a5b-69af-45b1-824f-a9dcb22a8c8b.png)

##### Sự ảnh hưởng của num_speaker với views  => Video có số lượng speaker càng ít thì thành tích (performance) có khả năng càng cao
plt.figure(figsize =(20,6))

plt.subplot(121)
sns.barplot( data = df.groupby('num_speaker')['views'].mean().reset_index(),
            x = 'num_speaker',
            y = 'views')

plt.subplot(122)
sns.boxplot(data = df,
            x = 'num_speaker',
            y = 'views')
plt.ylim(0,2e7)

![image](https://user-images.githubusercontent.com/103476246/169678244-8f43fafe-aa20-456e-a231-83c416b37a62.png)

##### Peak-time for filming video: Tháng 2, Tháng 6-7, Tháng 10-11
##### Peak-time for publishing video: tháng 4, tháng 9-10
##### thời gian publishing và filming được timing xen kẽ nhau khá rõ ràng

plt.figure(figsize = (20,6))

film_month = df.groupby('film_month').count()[['title']].reset_index()
sns.lineplot(data = film_month , x = 'film_month', y = 'title' , color = 'red')#.legend(['filmed'])
plt.ylabel('number of videos')
plt.xticks(ticks = range(1,13,1))

plt.twinx()

published_month = df.groupby('published_month').count()[['title']].reset_index()
sns.lineplot(data = published_month , x = 'published_month', y = 'title' )#.legend(['published'])
plt.ylabel('number of videos')
plt.xticks(ticks = range(1,13,1))

plt.legend(['filmed', 'published'])

![image](https://user-images.githubusercontent.com/103476246/169678276-7152a81e-fea9-4161-a4d7-ce06ab73b7d0.png)

##### Weekday nào có nhiều lượt xem nhất
##### Số lượng video được publish nhiều nhât vào đầu tuần và giảm dần về cuối tuần nhưng những video được chiếu có views trung bình cao nhất lại rơi vào giữa tuần
weekday_order = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
plt.figure(figsize=(20,6))
sns.lineplot(data =df.groupby('published_weekday').count()[['title']].loc[weekday_order],
             x = 'published_weekday',
             y= 'title',
             color = 'red')
plt.ylabel('Number of videos published')

plt.twinx()

sns.lineplot(data =df.groupby('published_weekday')['views'].mean().loc[weekday_order].reset_index(),
             x = 'published_weekday',
             y=  'views')

plt.legend(['Number of videos published', 'Average views'])

![image](https://user-images.githubusercontent.com/103476246/169678412-93f3ed15-8903-4999-9acb-158f78407a78.png)

##### Nghề nghiệp (speaker_occupation) nào popular, có thành tích tốt nhất 
##### Tuy writer được mời nhiều nhất nhưng psychology mới cho thấy sự hiệu quả trong việc thu hút audience khi có thành tích cao nhất
##### Director nên mời psychologist nhiều hơn, vì họ có nhiều outliers, mean cao nhất

plt.figure(figsize = (20,6))

plt.subplot(121)
occ_popular = df['speaker_occupation'].value_counts().reset_index().head(10)
sns.barplot(data = occ_popular,
            x = 'speaker_occupation',
            y = 'index')
plt.xlabel('Number of appearance')
plt.ylabel('Speaker_occupation')

plt.subplot(122)
sns.boxplot(data = df[df['speaker_occupation'].isin(occ_popular['index'])],
            x = 'speaker_occupation',
            y = 'views',
            order =occ_popular['index'] )
plt.ylim(0,1.75e7)

![image](https://user-images.githubusercontent.com/103476246/169678439-4a8471fc-cc53-4f7c-9881-8cddc0084bd4.png)

##### Đi sâu vào các tag của psychologist

df[df['speaker_occupation'] == 'Psychologist']['tags']

![image](https://user-images.githubusercontent.com/103476246/169678473-a691200f-67f9-4098-a379-7a949be8a665.png)

##### Thể loại (Tag) phổ biến nhất ở Tedtalk.
dummies = df['tags'].str.get_dummies(sep = ',')
sns.barplot(data = dummies.sum().reset_index().sort_values(0, ascending = False).head(10),
            x = 0,
            y = 'index')
plt.xlabel('appearance')
plt.ylabel('tag_name')

![image](https://user-images.githubusercontent.com/103476246/169678488-e4dc689f-43a6-4be2-8b72-3f63ac6c5459.png)

##### Những video có tỉ lệ engage cao thường sẽ có thể loại sensitive mang tính quan điểm như là culture, issue, religion, global, politics (comment/views)

top_engage = df.sort_values('engage_rate', ascending = False).head(10)
dummies = top_engage['tags'].str.get_dummies(sep = ',')
sns.barplot(data = dummies.sum().sort_values(ascending = False).head(5).reset_index(),
            x = 'index',
            y= 0)
plt.xlabel('appearance')
plt.ylabel('tag_name')

![image](https://user-images.githubusercontent.com/103476246/169678518-a4898319-b48c-490c-944d-30a8df1ce1a1.png)

### PROVE Negative is NOT BAD: Still Attract More Audience

##### Correlation between positive_rating, negative_rating and views
##### Những video có nhiều lượt đánh giá tích cực có view cao
##### Nhưng những videos có nhiều lượt đánh giá tiêu cực chưa chắc view thấp

plt.figure(figsize = (20,6))

plt.subplot(121)
sns.scatterplot(data = df,
                x = 'negative_rating',
                y = 'views')
plt.title('views - negative_rating')

plt.subplot(122)
sns.scatterplot(data = df,
                x = 'positive_rating',
                y = 'views')
plt.title('views - positive_rating')

![image](https://user-images.githubusercontent.com/103476246/169678559-1082f8d9-f614-4312-a352-08a7f2f4b4da.png)

##### Medium correlation between negative_rating and views 

df[['negative_rating', 'views']].corr()

![image](https://user-images.githubusercontent.com/103476246/169678568-e3704452-a1e8-4756-9e9e-2ac528155f0d.png)

##### Strong correlation between positive_rating and views

df[['positive_rating', 'views']].corr()

![image](https://user-images.githubusercontent.com/103476246/169678578-d69ce1b9-faaf-4eaa-b595-7e13c2120470.png)

##### Video có negative/positive ratio > 0.5 được đánh giá là những negative video

negative = df[df['negative/positive ratio'] >= 0.5].sort_values('negative/positive ratio', ascending = False)
other = df[df['negative/positive ratio'] < 0.5].sort_values('negative/positive ratio', ascending = False)

negative[['title', 'tags', 'ratings', 'negative/positive ratio', 'views', 'type']]

##### Phần trăm của những video million-view và under-million-view trên những video tiêu cực => tỉ lệ là 24.44% (104/230 * 100.0) 
##### 1/4 videos tiêu cực là million-view-video

pie = (negative['type'].value_counts() / negative.shape[0]).reset_index()
colors = sns.color_palette('pastel')[0:5]
plt.pie(pie['type'], labels = pie['index'],colors = colors, autopct='%.0f%%')

![image](https://user-images.githubusercontent.com/103476246/169678643-a33fae51-d7d8-4dbd-993e-79046cd3f639.png)

##### HOW to improve these negative to get the better performance?
##### Những video tiêu cực không tệ như chúng ta nghĩ vậy chúng ta có thể làm gì để cải thiện performance của chúng?
##### Phải tìm hiểu nguyên nhân

##### Tách các rating tiêu cực ra để xem những yếu tố nào audience đánh giá tiêu cực

negative_rating = ['Longwinded','Confusing','Unconvincing','Obnoxious'] 

for i in negative_rating:
    negative[i] = df['ratings'].apply(lambda x: int(x[i]))
    
##### Vậy là những video tiêu cực đa số sẽ được đánh giá là Unconvincing, Obnoxious

sns.barplot(data = negative[['Longwinded','Confusing','Unconvincing','Obnoxious']].sum().reset_index(),
            x = 0,
            y = 'index')
plt.xlabel('number of rating')
plt.ylabel('rating name')    

![image](https://user-images.githubusercontent.com/103476246/169678727-e0c14be8-b71e-41c6-9265-0e91f51550c8.png)

plt.figure(figsize= (20,6))
sns.heatmap(data= negative.corr(), annot=True, fmt='.2f')

##### Tuy audience đánh giá unconvincing nhiều nhất nhưng video bị đánh giá obnoxious mới engage được audience nhiều nhất

negative[['Longwinded','Confusing','Unconvincing', 'Obnoxious','engage_rate']].corr()

![image](https://user-images.githubusercontent.com/103476246/169678757-8e4a7477-8cc0-4f41-9dd9-8081f73da8d8.png)

##### Xem % Unconvincing của top 20 thể loại => tech, design, culture là 3 thể loại bị đánh giá không thuyết phục nhiều nhất
plt.figure(figsize = (20,6))
uncon_df = {
    'technology': negative[negative['tags'].str.contains('technology')]['Unconvincing'].sum(),
    'design': negative[negative['tags'].str.contains('design')]['Unconvincing'].sum(),
    'culture':negative[negative['tags'].str.contains('culture')]['Unconvincing'].sum(),
    'global issues':negative[negative['tags'].str.contains('global issues')]['Unconvincing'].sum(),
    'entertainment':negative[negative['tags'].str.contains('entertainment')]['Unconvincing'].sum(),
    'cities':negative[negative['tags'].str.contains('cities')]['Unconvincing'].sum(),
    'science':negative[negative['tags'].str.contains('science')]['Unconvincing'].sum(),
    'health':negative[negative['tags'].str.contains('health')]['Unconvincing'].sum(),
    'art':negative[negative['tags'].str.contains('art')]['Unconvincing'].sum(),
    'TEDx':negative[negative['tags'].str.contains('TEDx')]['Unconvincing'].sum(),
    'entertainment':negative[negative['tags'].str.contains('entertainment')]['Unconvincing'].sum(),  
    'innovation':negative[negative['tags'].str.contains('innovation')]['Unconvincing'].sum(),    
    'health':negative[negative['tags'].str.contains('health')]['Unconvincing'].sum(),          
    'society':negative[negative['tags'].str.contains('society')]['Unconvincing'].sum(),         
    'social change':negative[negative['tags'].str.contains('social change')]['Unconvincing'].sum(),    
    'business':negative[negative['tags'].str.contains('business')]['Unconvincing'].sum(),         
    'future':negative[negative['tags'].str.contains('future')]['Unconvincing'].sum(),           
    'humanity':negative[negative['tags'].str.contains('humanity')]['Unconvincing'].sum(),         
    'communication':negative[negative['tags'].str.contains('communication')]['Unconvincing'].sum(),   
    'environment':negative[negative['tags'].str.contains('environment')]['Unconvincing'].sum(),     
    'medicine':negative[negative['tags'].str.contains('medicine')]['Unconvincing'].sum(),       
    'business':negative[negative['tags'].str.contains('business')]['Unconvincing'].sum(),         
    'creativity':negative[negative['tags'].str.contains('creativity')]['Unconvincing'].sum(),      
    'collaboration':negative[negative['tags'].str.contains('collaboration')]['Unconvincing'].sum()    
}
sns.barplot(data = pd.Series(uncon_df).sort_values(ascending = False).reset_index(), x ='index', y = 0)

![image](https://user-images.githubusercontent.com/103476246/169678814-9d647079-2085-4476-9951-d345b49d9e48.png)

##### Đào sâu vào 3 tag nó có %convincing cao

tech_des_cul = negative[negative['tags'].str.contains('technology|design|culture|global issues|science')]

##### Nghề nghiệp của những speakers trong 3 thể loại bị đánh giá tiêu cực Unconvincing cao.

job_count = []
for i in tech_des_cul['speaker_occupation']:
    job_count.append(i)
pd.Series(job_count).value_counts().reset_index().head()

![image](https://user-images.githubusercontent.com/103476246/169678858-9f72261e-2013-4110-87c7-d7a7949dea77.png)

##### Đào sâu phân tích tiếp event, obnoxious

tech_des_cul[['speaker_occupation', 'tags', 'languages', 'views', 'Unconvincing']]

![image](https://user-images.githubusercontent.com/103476246/169678897-6a0983f9-d413-471c-b9f1-df259c3866e7.png)

##### Thể loại phổ biến ở những video tiêu cực
##### Technology vẫn giữ nguyên vị trí là một trong những video unconvincing. Đặc biệt, design (4 lên 2) và entertainment (7 lên 5)
##### đã vượt lên 2 hạng => 2 tag này khả nghi trong việc làm tăng đánh giá tiêu cực

plt.figure(figsize =(20,6))

plt.subplot(121)
dummies = negative['tags'].str.get_dummies(sep = ',')
dummies_plot = dummies.sum().reset_index().sort_values(0, ascending = False).head(10)
sns.barplot(data = dummies_plot,
            x = 0,
            y = 'index')
plt.xlabel('appearance')
plt.ylabel('tag_name')
plt.title('Top 10 popular Tag in negative videos')

plt.subplot(122)
dummies = df['tags'].str.get_dummies(sep = ',')
sns.barplot(data = dummies.sum().reset_index().sort_values(0, ascending = False).head(10),
            x = 0,
            y = 'index')
plt.xlabel('appearance')
plt.ylabel('tag_name')
plt.title('Top 10 popular Tag')

![image](https://user-images.githubusercontent.com/103476246/169678934-3bba706d-d621-4895-b230-db7acf802530.png)

dummies_plot['index']

![image](https://user-images.githubusercontent.com/103476246/169678949-a7903d0f-fa75-49cc-a65d-1414b9e1298c.png)

##### Thành tích của top 10 thể loại popular ở những video tiêu cực (không cần thiết)

tmp = pd.concat([negative,dummies], axis = 1)
clean_tmp = tmp.melt(id_vars = tmp.columns[:10],
                     value_vars = tmp.columns[11:],
                     var_name ='tag_name',
                     value_name = 'Yes')
clean_tmp = clean_tmp[clean_tmp['Yes'] == 1]
clean_tmp[clean_tmp['tag_name'].isin(dummies_plot['index'])]

plt.figure(figsize = (20,6))
sns.boxplot(data = clean_tmp[clean_tmp['tag_name'].isin(dummies_plot['index'])],
            x = 'tag_name',
            y = 'views')

##### Đúng vậy, Video tiêu cực đa số được trình bày bởi những diễn giả có occupation liên quan đến design (architect, designer) và entertainment(artist, musician)

sns.barplot(data = negative['speaker_occupation'].value_counts().head().reset_index(),
            x = 'speaker_occupation',
            y = 'index')
plt.ylabel('speaker_occupation')
plt.xlabel('Number of videos')

![image](https://user-images.githubusercontent.com/103476246/169678975-d6f338a0-cd7f-41fd-961e-95326cfba63e.png)

##### Video bị đánh giá tiêu cực(negative/positive ratio > 0.5) không hẳn là XẤU.
##### Trong đó, million-view videos chiếm 24,44%, những non-million videos có view trung bình là 528k (cao nhất 940k, thấp nhất 176k).
##### Thể loại thiên về creativity (Design, Tech, Global, Culture, Issues), có sự xuất hiện của Design (top 1: 21 videos).
##### Rating bị đánh giá đa số Unconvincing
##### Speaker_occupation là những ngành liên quan đến sáng tạo (Artist, Architect, Philosopher, Musician, Designer) có thể khả năng trình bày của họ không được tốt như writer
#####  Sau khi phân tích ở những video bị đánh giá tiêu cực: director không nên lo lắng về những video bị đánh giá tiêu cực,
##### Vì những speaker đều có background liên quan đến creativity nên có thể khả năng trình bày chưa được tốt.
##### Tập trung cải thiện chất lượng ở videos thể loại Design tiêu biểu là cải thiện việc cách trình bày của speaker(vì audience đánh giá nhiều là Unconvincing)
