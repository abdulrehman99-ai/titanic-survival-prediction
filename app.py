
import streamlit as st
import pandas as pd
import joblib

# 1. Page settings
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# 2. Load trained model
model = joblib.load("titanic_survival_model.pkl")

# 3. App title
st.title("🚢 Titanic Survival Prediction")
st.write(
    "Passenger ki details enter karein aur survival prediction hasil karein."
)

st.divider()

# 4. User input
st.subheader("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        index=2
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Siblings / Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

with col2:
    parch = st.number_input(
        "Parents / Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        value=32.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        }[x]
    )

st.divider()

# 5. Prediction button
if st.button("Predict Survival", type="primary", use_container_width=True):

    # Input data ko DataFrame mein convert karna
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    # Model prediction
    prediction = model.predict(input_data)[0]

    # Estimated survival probability
    probabilities = model.predict_proba(input_data)[0]
    survival_probability = probabilities[list(model.classes_).index(1)] * 100

    # 6. Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Passenger ke survive karne ka imkaan hai.")
    else:
        st.error("Prediction: Passenger ke survive na karne ka imkaan hai.")

    st.metric(
        "Estimated Survival Probability",
        f"{survival_probability:.2f}%"
    )

    st.progress(int(round(survival_probability)))

    st.caption(
        "Note: Ye trained model ki prediction hai, actual outcome ki guarantee nahi."
    )
