import streamlit as st
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import pickle
import requests
from streamlit_lottie import st_lottie
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt




cat_features = ["Term", "Years in current job", "Home Ownership", "Purpose", "Number of Credit Problems",
                "Bankruptcies", "Tax Liens"]
num_features = ['Current Loan Amount', 'Annual Income', 'Monthly Debt', 'Current Credit Balance',
                'Maximum Open Credit', "Credit Score", "Years of Credit History", "Number of Open Accounts",
                "Months since last delinquent"]


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Streamlit app code
with st.container():
    left, right = st.columns(2)
    with left:
        st.title(":red[Predictive Mortgage Payment Evaluation with Interactive Platform]")
        st.write("""
        **********************************************
        *            Mortgage Debt Predictor          *
        **********************************************

        Welcome to Mortgage Debt Predictor, your 
        reliable companion in financial forecasting.
        
        - Project Overview -
        
        Our Machine Learning endeavor revolves around
        predicting whether a customer will fulfill 
        their mortgage obligations. We've analyzed 
        diverse customer attributes and explored 
        various models, from Decision Trees to 
        Logistic Regression.
        
        - Model Selection -
        
        After rigorous tuning and testing, AdaBoost
        emerged as our champion. This model, finely 
        tuned via GridSearch, boasts an impressive 
        accuracy of 86.97%, precision of 100%, and 
        recall of 69%.
        
        - Interactive Interface -
        
        To make predictions accessible, we've crafted 
        an interactive GUI. Users can effortlessly 
        input key features and obtain predictions 
        regarding mortgage repayment, facilitating 
        informed decision-making.
        
        Join us on this journey towards financial 
        foresight!
        
        **********************************************
        """)
    with right:
        lottie_url = " https://lottie.host/0caf0142-8476-4755-bef1-4a166b67f6a3/vNFugFbq57.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=400)

        lottie_url = "https://lottie.host/9b1d760a-d152-4817-9a7c-d5dce70d0f96/65tWrCArzp.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=200)

        lottie_url = "https://lottie.host/d27c410d-c34e-494c-826b-47d37805e1e1/VkSmAWhA8B.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=400)

df_train = pd.read_csv("dataset.csv")
df_train1= pd.read_csv("credit_train.csv")
# Obtener todas las columnas de df_test excepto "LOAN STATUS" para X
X = df_train1.drop(columns=["Loan Status", "Customer ID", "Loan ID"])

# Obtener la columna "LOAN STATUS" para y
y = df_train1["Loan Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df_train.dropna(inplace=True)

num_input = {}
cat_input = {}
st.title(":blue[Feature Selection App]")


with st.container():
    st.subheader(":red[Selección de features] :wave:")
    left, right = st.columns(2)
    with left:
        for feature in cat_features:
            value = st.selectbox(f':blue[{feature}]', df_train[feature].unique(), key=feature)
            cat_input[feature] = value

    with right:
        for feature in num_features:
            min_val = int(df_train[feature].min())
            max_val = int(df_train[feature].max())
            selected_value = st.slider(f':blue[{feature}]', min_value=min_val, max_value=max_val, value=min_val)
            num_input[feature] = selected_value

    # Add a button for prediction
    if st.button("Make Prediction", key="prediction_button"):
        user_input = {}
        user_input.update(num_input)
        user_input.update(cat_input)

        # Load the model from the file
        with open('models/pipeline_AdaBoost.pkl', 'rb') as file:
            model = pickle.load(file)

        prediction = model.predict(pd.DataFrame(user_input, index=[0]))
        
        if prediction == 1:

            st.write(f":red[No paga la deuda]")
        else:
            st.write(f":green[Paga la deuda]")

with open('models/pipeline_AdaBoost.pkl', 'rb') as file:
            model = pickle.load(file)



# Load your dataset
# df_train = pd.read_csv('your_dataset.csv')

# Set up subplots
fig = sp.make_subplots(rows=6, cols=3, subplot_titles=("Loan Status", "Current Loan Amount", "Term",
                                                      "Credit Score", "Annual Income", "Years in Current Job",
                                                      "Home Ownership", "Purpose", "Monthly Debt",
                                                      "Years of Credit History", "Months Since Last Delinquent",
                                                      "Number of Open Accounts", "Number of Credit Problems",
                                                      "Current Credit Balance", "Maximum Open Credit",
                                                      "Bankruptcies", "Tax Liens"))

# Populate subplots
fig.add_trace(go.Histogram(x=df_train["Loan Status"]), row=1, col=1)
fig.add_trace(go.Histogram(x=df_train["Current Loan Amount"]), row=1, col=2)
fig.add_trace(go.Histogram(x=df_train["Term"]), row=1, col=3)
fig.add_trace(go.Histogram(x=df_train["Credit Score"]), row=2, col=1)
fig.add_trace(go.Histogram(x=df_train["Annual Income"]), row=2, col=2)
fig.add_trace(go.Histogram(y=df_train["Years in current job"]), row=2, col=3)
fig.add_trace(go.Histogram(x=df_train["Home Ownership"]), row=3, col=1)
fig.add_trace(go.Histogram(y=df_train["Purpose"]), row=3, col=2)
fig.add_trace(go.Histogram(x=df_train["Monthly Debt"]), row=3, col=3)
fig.add_trace(go.Histogram(x=df_train["Years of Credit History"]), row=4, col=1)
fig.add_trace(go.Histogram(x=df_train["Months since last delinquent"]), row=4, col=2)
fig.add_trace(go.Histogram(x=df_train["Number of Open Accounts"]), row=4, col=3)
fig.add_trace(go.Histogram(x=df_train["Number of Credit Problems"]), row=5, col=1)
fig.add_trace(go.Histogram(x=df_train["Current Credit Balance"]), row=5, col=2)
fig.add_trace(go.Histogram(x=df_train["Maximum Open Credit"]), row=5, col=3)
fig.add_trace(go.Histogram(x=df_train["Bankruptcies"]), row=6, col=1)
fig.add_trace(go.Histogram(x=df_train["Tax Liens"]), row=6, col=2)

# Update layout
fig.update_layout(title_text="Distribucion Variables", showlegend=False, height=1200, width=1000)

# Display the plot
st.plotly_chart(fig)
