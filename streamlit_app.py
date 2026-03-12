
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Настройка страницы
st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("Прогноз оттока клиентов банка")

# 2. Загрузка модели
@st.cache_resource
def load_model():
    # Файл xgboost_churn_model.joblib должен быть в корне репозитория GitHub
    return joblib.load('xgboost_churn_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")

st.sidebar.header("Данные клиента")

# 3. Функция сбора данных
def user_input_features():
    # Определяем маппинги ВНУТРИ функции до их использования
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Female": 0, "Male": 1}
    card_map = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}

    # Поля ввода в сайдбаре
    geography = st.sidebar.selectbox("География", list(geo_map.keys()))
    gender = st.sidebar.selectbox("Пол", list(gender_map.keys()))
    age = st.sidebar.slider("Возраст", 18, 100, 35)
    credit_score = st.sidebar.slider("Кредитный рейтинг", 300, 850, 600)
    tenure = st.sidebar.slider("Лет в банке", 0, 10, 5)
    balance = st.sidebar.number_input("Баланс", value=0.0)
    num_products = st.sidebar.selectbox("Кол-во продуктов", (1, 2, 3, 4))
    has_card = st.sidebar.checkbox("Есть кредит. карта", value=True)
    is_active = st.sidebar.checkbox("Активный клиент", value=True)
    salary = st.sidebar.number_input("Зарплата", value=50000.0)
    
    # Новые поля, которые требовала модель
    complain = st.sidebar.checkbox("Были жалобы", value=False)
    satisfaction = st.sidebar.slider("Удовлетворенность", 1, 5, 3)
    card_type = st.sidebar.selectbox("Тип карты", list(card_map.keys()))
    points = st.sidebar.number_input("Баллы", value=0)

    # Создаем словарь в ТОМ ЖЕ ПОРЯДКЕ, в котором обучалась модель
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
        'EstimatedSalary': salary,
        'Complain': int(complain),
        'Satisfaction Score': satisfaction,
        'Card Type': card_map[card_type],
        'Point Earned': points
    }
    return pd.DataFrame(data, index=[0])

# Вызов функции
input_df = user_input_features()

st.subheader("Параметры для анализа")
st.write(input_df)

# 4. Прогноз
if st.button("Рассчитать риск"):
    try:
        # Получаем имена признаков, которые реально ждет модель
        feature_names = model.get_booster().feature_names
        # Сортируем колонки нашего ввода точно по модели
        input_ready = input_df[feature_names]
        
        prediction = model.predict(input_ready)
        probability = model.predict_proba(input_ready)[0][1]
        
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ **Высокий риск ухода!**")
            st.write(f"Вероятность: **{probability:.2%}**")
        else:
            st.success(f"✅ **Клиент лоялен.**")
            st.write(f"Вероятность ухода: **{probability:.2%}**")
            
    except Exception as e:
        st.error(f"Ошибка при расчете: {e}")
