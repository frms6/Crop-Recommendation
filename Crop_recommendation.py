# %load Crop_recommendation.py
"""
Streamlit app for AI Predict
"""
import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
# --- page
st.set_page_config(page_title='Predict', page_icon=':heart:')


@st.cache_data
def data_preprocess(csv):
    Crop_recommendation = pd.read_csv(csv, encoding='utf8')


    return Crop_recommendation

#@st.cache_resource
def load_ai_pickle(pkl):
    import joblib
    dct = joblib.load(pkl)
    return dct

begin = time.time()

BASE_DIR = Path.cwd()
CSV_FILE =  BASE_DIR / 'Crop_recommendation.csv'
PICKLE_NAME = BASE_DIR / 'Crop_recommendation.pkl'

# --- data prepare
Crop = data_preprocess(CSV_FILE)
dctPickle = load_ai_pickle(PICKLE_NAME)

# --- Define Crop Name Mapping ---
crop_mapping = {
    'rice': '쌀',
    'maize': '옥수수',
    'chickpea': '병아리콩',
    'kidneybeans': '강낭콩',
    'pigeonpeas': '비둘기콩',
    'mothbeans': '나방콩',
    'mungbean': '녹두',
    'blackgram': '검은콩(흑태)',
    'lentil': '렌틸콩',
    'pomegranate': '석류',
    'banana': '바나나',
    'mango': '망고',
    'grapes': '포도',
    'watermelon': '수박',
    'muskmelon': '멜론',
    'apple': '사과',
    'orange': '오렌지',
    'papaya': '파파야',
    'coconut': '코코넛',
    'cotton': '목화',
    'jute': '황마',
    'coffee': '커피'
}

# --- sidebar
# 토양 정보

N_min, N_max, N_mean = int(Crop.N.min()), int(Crop.N.max()), int(Crop.N.mean())
P_min, P_max, P_mean = int(Crop.P.min()), int(Crop.P.max()), int(Crop.P.mean())
K_min, K_max, K_mean = int(Crop.K.min()), int(Crop.K.max()), int(Crop.K.mean())

# 기후 정보
temperature_min, temperature_max, temperature_mean = int(Crop.temperature.min()), int(Crop.temperature.max()), int(Crop.temperature.mean())
humidity_min, humidity_max, humidity_mean = int(Crop.humidity.min()), int(Crop.humidity.max()), int(Crop.humidity.mean())
ph_min, ph_max, ph_mean = int(Crop.ph.min()), int(Crop.ph.max()), int(Crop.ph.mean())
rainfall_min, rainfall_max, rainfall_mean = int(Crop.rainfall.min()), int(Crop.rainfall.max()), int(Crop.rainfall.mean())

# 토양 정보 선택
N = st.sidebar.slider("질소(kg/ha)", N_min, N_max, N_mean, 1)
P = st.sidebar.slider("인(kg/ha)", P_min, P_max, P_mean, 1)
K = st.sidebar.slider("칼륨(kg/ha)", K_min, K_max, K_mean, 1)

# 기후 정보 선택
temperature = st.sidebar.slider("기온(℃)", temperature_min, temperature_max, temperature_mean, 1)
humidity = st.sidebar.slider("습도(%)", humidity_min, humidity_max, humidity_mean, 1)
ph = st.sidebar.slider("pH", ph_min, ph_max, ph_mean, 1)
rainfall = st.sidebar.slider("강우량(mm)", rainfall_min, rainfall_max, rainfall_mean, 1)

# --- body
#st.title("작물 추천")
#st.write("#### 환경에 맞는 작물을 추천합니다.")
#st.write("---")

st.markdown("## 🌾 **작물 추천 도구**")
st.write("#### 다음 환경 조건에 맞는 추천 작물입니다:")
st.markdown("---")  # Add a separator line


widgetData = [N, P, K, temperature, humidity, ph, rainfall]

dfPredict = pd.DataFrame(data=[widgetData], columns=Crop.columns[:-1])
st.write(dfPredict)

pred = dctPickle.predict(dfPredict)
key = pred[0]
crop = crop_mapping[key]

st.markdown("###  **추천 작물**")
st.markdown(f"#### **{crop}**")

