import streamlit as st
from pages import home, data_explorer, model_training, fraud_detection

PAGES = {
    "Home": home,
    "Data Explorer": data_explorer,
    "Model Training": model_training,
    "Fraud Detection": fraud_detection
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.app()

if __name__ == "__main__":
    main()