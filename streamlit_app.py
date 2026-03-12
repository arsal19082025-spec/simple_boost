
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
    
   # НОВЫЕ ПОЛЯ:
    complain = st.sidebar.checkbox("Были ли жалобы (Complain)?", value=False)
    satisfaction = st.sidebar.slider("Уровень удовлетворенности (Satisfaction Score)", 1, 5, 3)
    card_type = st.sidebar.selectbox("Тип карты (Card Type)", ("DIAMOND", "GOLD", "SILVER", "PLATINUM"))
    points = st.sidebar.number_input("Накопленные баллы (Point Earned)", value=0)
    
    # Маппинг для типа карты (должен совпадать с тем, что сделал LabelEncoder в ноутбуке)
    # Если вы использовали LabelEncoder для 'Card Type', порядок обычно алфавитный:
    card_map = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}
    
    # Формируем итоговый словарь со ВСЕМИ колонками в правильном порядке
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

input_df = user_input_features()

st.subheader("Параметры для анализа")
st.write(input_df)

# Прогноз
if st.button("Рассчитать риск"):
    try:
        # 1. Получаем список имен признаков, которые ожидает модель
        expected_features = model.get_booster().feature_names
        
        # 2. Переупорядочиваем колонки в input_df в соответствии с моделью
        # Если каких-то колонок не хватает, это вызовет понятную ошибку
        input_df_reshaped = input_df[expected_features]
        
        prediction = model.predict(input_df_reshaped)
        probability = model.predict_proba(input_df_reshaped)[0][1]
        
        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ **Высокий риск ухода!**")
            st.write(f"Вероятность: **{probability:.2%}**")
        else:
            st.success(f"✅ **Клиент лоялен.**")
            st.write(f"Вероятность ухода: **{probability:.2%}**")
            
    except KeyError as e:
        st.error(f"Ошибка: Модель ожидает признак {e}, которого нет в форме ввода.")
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")


