
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
    # Файл должен лежать в корне репозитория на GitHub
    return joblib.load('xgboost_churn_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}. Убедитесь, что файл .joblib загружен в GitHub.")

st.sidebar.header("Данные клиента")

def user_input_features():
    geography = st.sidebar.selectbox("География", ("France", "Germany", "Spain"))
    gender = st.sidebar.selectbox("Пол", ("Male", "Female"))
    age = st.sidebar.slider("Возраст", 18, 100, 35)
    credit_score = st.sidebar.slider("Кредитный рейтинг", 300, 850, 600)
    
    # Добавленные поля, чтобы модель не выдавала ошибку:
    tenure = st.sidebar.slider("Сколько лет с банком (Tenure)", 0, 10, 5)
    balance = st.sidebar.number_input("Баланс на счету", value=0.0)
    num_products = st.sidebar.selectbox("Кол-во продуктов", (1, 2, 3, 4))
    has_card = st.sidebar.checkbox("Есть кредитная карта?", value=True)
    is_active = st.sidebar.checkbox("Активный клиент?", value=True)
    salary = st.sidebar.number_input("Предполагаемая зарплата", value=50000.0)
    
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Female": 0, "Male": 1}
    
    data = {
        'CreditScore': credit_score,
        'Geography': geo_map[geography],
        'Gender': gender_map[gender],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_card),
        'IsActiveMember': int(is_active),
        'EstimatedSalary': salary
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Параметры для анализа")
st.write(input_df)

# Прогноз
if st.button("Рассчитать риск"):
    # Делаем копию, чтобы не испортить исходный DF
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **Высокий риск ухода!**")
        st.write(f"Вероятность: **{probability:.2%}**")
    else:
        st.success(f"✅ **Клиент лоялен.**")
        st.write(f"Вероятность ухода: **{probability:.2%}**")

