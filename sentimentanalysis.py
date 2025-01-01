from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

st.header('Sentiment Analysis')
st.write('In sentiment analysis, there are two most important components to classify a sentence, those are polarity and subjectivity.')

st.divider()

st.header('Analyze Text')
st.write('Polarity is the overall sentiment of a text, phrase, or word, and is an important part of the process of classifying sentiment. It is used to determine if the sentiment expressed in a text is positive, negative, or neutral. It is often expressed as a numerical rating, or sentiment score, that ranges from -1 (very negative) to +1 (very positive). The score can be calculated for an entire text or for individual phrases.')
st.write('Subjectivity is the task of classifying a sentence as opinionated or not opinionated. The resulting opinionated sentences are also classified as expressing positive or negative opinions, which is called the sentence- level sentiment classification. It is often expressed as a float value in the range from 0 to 1, where 0 represents objective text and 1 represents highly subjective text.')
st.write('')

text = st.text_input('Input text to analyze sentiment score: ')
if text:
    blob = TextBlob(text)
    st.write(f'Polarity: {blob.sentiment.polarity}')
    st.write(f'Subjectivity: {blob.sentiment.subjectivity}')

pre = st.text_input('Input text to be cleaned: ')
if pre:
    st.write('Cleaned text:')
    st.write(cleantext.clean(
        pre, clean_all= False, extra_spaces=True, 
        stopwords=True, lowercase=True, numbers=True, punct=True))

st.divider()

st.header('Analyze CSV')
st.write('Make sure the file that you uploaded is in CSV format, otherwise it would appear an error message.')
uploaded = st.file_uploader('Upload a CSV file')

def clean_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''  # Return an empty string if the input is not a valid string
    return cleantext.clean(
        text, clean_all=False, extra_spaces=True,
        stopwords=True, lowercase=True, numbers=True, punct=True
    )

def score(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity

def analyze(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

if uploaded:
    df = pd.read_csv(uploaded)

    # Ensure the text column is string and handle null values
    df['text'] = df['text'].fillna('').astype(str)
        
    # Clean the text column
    df['text'] = df['text'].apply(clean_text)
        
    # Calculate the polarity score
    df['score'] = df['text'].apply(score)
        
    # Determine sentiment
    df['sentiment'] = df['score'].apply(analyze)
        
    # Display the dataframe
    st.write(df[['text', 'sentiment', 'score']].head(100))

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    # Convert the dataframe to CSV
    csv = convert_df(df)

    # Download button for the cleaned and analyzed data
    st.download_button(
        label="Download data as CSV", data=csv,
        file_name='sentiment.csv', mime='text/csv',
    )

# streamlit run sentimentanalysis.py [-- script args]
