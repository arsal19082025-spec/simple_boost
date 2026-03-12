
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("Прогноз оттока клиентов банка")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('xgboost_churn_model.joblib')

model = load_model()

st.sidebar.header("Данные клиента")

def user_input_features():
    # Создаем поля ввода на основе признаков вашего датасета
    geography = st.sidebar.selectbox("География", ("France", "Germany", "Spain"))
    gender = st.sidebar.selectbox("Пол", ("Male", "Female"))
    age = st.sidebar.slider("Возраст", 18, 100, 35)
    credit_score = st.sidebar.slider("Кредитный рейтинг", 300, 850, 600)
    balance = st.sidebar.number_input("Баланс на счету", value=0.0)
    num_products = st.sidebar.selectbox("Кол-во продуктов", (1, 2, 3, 4))
    is_active = st.sidebar.checkbox("Активный клиент?", value=True)
    
    # Маппинг категорий (должен совпадать с тем, как обучали модель!)
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Female": 0, "Male": 1}
    
    data = {
        'CreditScore': credit_score,
        'Geography': geo_map[geography],
        'Gender': gender_map[gender],
        'Age': age,
        'Balance': balance,
        'NumOfProducts': num_products,
        'IsActiveMember': int(is_active)
        # Добавьте остальные колонки, если они использовались при обучении
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Параметры для анализа")
st.write(input_df)

# Прогноз
if st.button("Рассчитать риск"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ Высокий риск ухода! Вероятность: {probability:.2%}")
    else:
        st.success(f"✅ Клиент лоялен. Вероятность ухода: {probability:.2%}")
