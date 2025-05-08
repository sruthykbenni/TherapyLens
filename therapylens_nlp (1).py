# STEP 2: Import Libraries
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from bertopic import BERTopic
from langdetect import detect
import nltk
nltk.download('punkt')

# STEP 3: Upload Dataset
uploaded_file = st.file_uploader("/content/intents.json", type="json")


# STEP 4: Load and Prepare Data
import io
data = json.load(io.StringIO(uploaded['intents.json'].decode('utf-8')))
entries = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        entries.append({'text': pattern, 'tag': intent.get('tag', 'unknown')})
df = pd.DataFrame(entries)
df['session'] = (df.index // 10) + 1

# STEP 5: Multilingual Detection (Optional Filtering)
from langdetect import detect
df['language'] = df['text'].apply(lambda x: detect(x))
df = df[df['language'] == 'en']  # Keeping only English entries

# STEP 6: Sentiment & Emotion Detection
sentiment_pipeline = pipeline("sentiment-analysis")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
df['sentiment_result'] = df['text'].apply(lambda x: sentiment_pipeline(x)[0])
df['sentiment'] = df['sentiment_result'].apply(lambda x: x['label'])
df['sentiment_score'] = df['sentiment_result'].apply(lambda x: x['score'])
df['emotion_result'] = df['text'].apply(lambda x: emotion_pipeline(x)[0][0])
df['emotion'] = df['emotion_result'].apply(lambda x: x['label'])
df['emotion_score'] = df['emotion_result'].apply(lambda x: x['score'])

# STEP 7: Emotion Intensity Categorization
def categorize_intensity(score):
    if score >= 0.85: return "Extreme"
    elif score >= 0.65: return "Strong"
    elif score >= 0.4: return "Moderate"
    else: return "Mild"
df['emotion_intensity'] = df['emotion_score'].apply(categorize_intensity)

# STEP 8: Topic Modeling with BERTopic
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(df['text'])
df['topic'] = topics

# STEP 9: Session Summarization with T5
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize(text):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=50, min_length=10, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

df['session_text'] = df.groupby('session')['text'].transform(lambda x: " ".join(x))
df['session_summary'] = df.groupby('session')['session_text'].transform(lambda x: summarize(x.iloc[0]) if len(x.iloc[0].split()) > 20 else "Too short for summary")

# STEP 10: Critical Emotion Alert System
def alert_flag(row):
    if row['emotion'] in ['anger', 'sadness'] and row['emotion_score'] > 0.8:
        return "ALERT"
    return "NORMAL"
df['alert'] = df.apply(alert_flag, axis=1)
df['consec_alert'] = df['alert'].eq('ALERT').astype(int)
df['consec_alert'] = df['consec_alert'].groupby(df['session']).transform('sum')
df['session_alert'] = df['consec_alert'].apply(lambda x: 'CRITICAL' if x >= 3 else 'OK')

# STEP 11: Visualizations
sentiment_trend = df.groupby("session")["sentiment"].value_counts().unstack().fillna(0)
emotion_trend = df.groupby("session")["emotion"].value_counts().unstack().fillna(0)

sentiment_trend.plot(title="Sentiment Trend Across Sessions", figsize=(10,5))
plt.grid(True)
plt.show()

emotion_trend.plot(title="Emotion Trend Across Sessions", figsize=(10,5))
plt.grid(True)
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['text']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Diary Entries")
plt.show()

# STEP 12: Export Final Results
df.to_csv("therapy_emotion_analysis_advanced.csv", index=False)
print("✅ Final results saved to 'therapy_emotion_analysis_advanced.csv'")
