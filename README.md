# Analyze-Data-TedTalk-in-Kaggle-Website
These datasets contain information about all audio-video recordings of TED Talks uploaded to the official TED.com website until September 21st, 2017. The TED main dataset contains information about all talks including number of views, number of comments, descriptions, speakers and titles. The TED transcripts dataset contains the transcripts for all talks available on TED.com

There are two CSV files.

ted_main.csv - Contains data on actual TED Talk metadata and TED Talk speakers.

transcripts.csv - Contains transcript and URL information for TED Talks

# 1-COLECT DATA
# Create the connection between google colab and google drive
from google.colab import drive
drive.mount('/content/drive')

# Attach the library
import pandas as pd
import numpy as np
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
import re

# Create the linking tables
main_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ted_main.csv')
trans_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/transcripts.csv')

# Merge the ted_main and transcripts 
df = pd.merge(main_df, trans_df, how = 'left', on = 'url')

# 2-CLEAN DATA 
# Dropping unnecessary columns including url and description in a transcripts file
# Changing the original DataFrame, use the inplace = True argument
df.drop(columns=['url'],inplace=True)

# Arrange order of columns
df = df[['name', 'title', 'main_speaker', 'num_speaker', 'speaker_occupation', 'duration', 'film_date', 'published_date', 'event', 'comments', 'tags', 'languages', 'ratings', 'views', 'transcript', 'description']]

# Change numerical format
pd.options.display.float_format = "{:.2f}".format

# Change the time of duration to minutes   
df['duration'] = round(df['duration']/60, 2)

# Change the time of film_date, and published_date from senconds to yyyy-mm-dd and hh:mm:ss
df['film_date'] = pd.to_datetime(df['film_date'], unit = 's')                  # df.film_date 
df['published_date'] = pd.to_datetime(df['published_date'], unit = 's')        # df.published_date

# Separate day, month, year of the columns of film_date & published_date
df['Year'] = df['published_date'].dt.year
df['Year_Month_filming']=df['film_date'].dt.to_period('M')
df['Year_Month_publishing']=df['published_date'].dt.to_period('M')

# Replace missing data by scalar value 
df['speaker_occupation'] = df['speaker_occupation'].fillna('Unknown')
df['transcript'] = df['transcript'].fillna('Unknown')

df.isnull().sum()

![image](https://user-images.githubusercontent.com/103476246/168977426-444d27d9-5dc6-4232-b727-5f29cbb3349f.png)





