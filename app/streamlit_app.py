import streamlit as st
import requests
from datetime import datetime


# ============================================================
# CREDIT CARD FRAUD DETECTION - STREAMLIT FRONTEND
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-fraud {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border: 2px solid #ff4b4b;
    }

    .result-safe {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border: 2px solid #21c354;
    }

    .probability {
        text-align: center;
        font-size: 22px;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Transaction Fraud Detection System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API STATUS
# ============================================================

try:

    health_response = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=3
    )

    if health_response.status_code == 200:

        st.success("🟢 Fraud Detection API is connected")

    else:

        st.warning(
            "⚠️ API is running but health check failed"
        )

except requests.exceptions.RequestException:

    st.error(
        "🔴 Cannot connect to FastAPI. "
        "Make sure the FastAPI server is running."
    )


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.markdown("## 📝 Enter Transaction Details")

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    transaction_id = st.text_input(
        "Transaction ID",
        value="TX10001"
    )

    customer_id = st.text_input(
        "Customer ID",
        value="CUST001"
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1500.0,
        step=100.0
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )

    previous_transactions = st.number_input(
        "Previous Transactions",
        min_value=0,
        value=10,
        step=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    timestamp = st.text_input(
        "Timestamp",
        value=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # IMPORTANT:
    # These values match the actual dataset.

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "fuel",
            "electronics",
            "entertainment",
            "fashion",
            "grocery"
        ]
    )

    customer_location = st.selectbox(
        "Customer Location",
        [
            "NY",
            "CA",
            "TX",
            "FL",
            "IL"
        ]
    )

    device_type = st.selectbox(
        "Device Type",
        [
            "mobile",
            "tablet",
            "desktop"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Fraud",
    use_container_width=True
)


# ============================================================
# SEND DATA TO FASTAPI
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create JSON request
    # --------------------------------------------------------

    transaction_data = {

        "transaction_id": transaction_id,

        "timestamp": timestamp,

        "amount": amount,

        "merchant_category": merchant_category,

        "customer_id": customer_id,

        "customer_age": customer_age,

        "customer_location": customer_location,

        "device_type": device_type,

        "previous_transactions": previous_transactions
    }


    # --------------------------------------------------------
    # Validate basic inputs
    # --------------------------------------------------------

    if transaction_id.strip() == "":

        st.error(
            "Please enter a Transaction ID."
        )

    elif customer_id.strip() == "":

        st.error(
            "Please enter a Customer ID."
        )

    elif amount <= 0:

        st.error(
            "Transaction amount must be greater than 0."
        )

    else:

        try:

            # ------------------------------------------------
            # Send request to FastAPI
            # ------------------------------------------------

            with st.spinner(
                "Analyzing transaction..."
            ):

                response = requests.post(
                    API_URL,
                    json=transaction_data,
                    timeout=10
                )


            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]

                probability = float(
                    result["fraud_probability"]
                )

                threshold = float(
                    result["threshold"]
                )


                # =================================================
                # FRAUD RESULT
                # =================================================

                if prediction == "FRAUD":

                    st.markdown(
                        """
                        <div class="result-fraud">
                        🚨 FRAUDULENT TRANSACTION DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # LEGITIMATE RESULT
                # =================================================

                else:

                    st.markdown(
                        """
                        <div class="result-safe">
                        ✅ LEGITIMATE TRANSACTION
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # ------------------------------------------------
                # Display probability
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="probability">
                    Fraud Probability:
                    <b>{probability * 100:.2f}%</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------------
                # Result metrics
                # ------------------------------------------------

                st.markdown(
                    "### 📊 Prediction Details"
                )

                metric1, metric2, metric3 = st.columns(3)


                with metric1:

                    st.metric(
                        "Transaction ID",
                        result["transaction_id"]
                    )


                with metric2:

                    st.metric(
                        "Customer ID",
                        result["customer_id"]
                    )


                with metric3:

                    st.metric(
                        "Detection Threshold",
                        f"{threshold:.2f}"
                    )


                # ------------------------------------------------
                # Probability progress bar
                # ------------------------------------------------

                st.markdown(
                    "### Fraud Probability"
                )

                st.progress(
                    min(
                        max(probability, 0.0),
                        1.0
                    )
                )


                # ------------------------------------------------
                # Explanation
                # ------------------------------------------------

                if prediction == "FRAUD":

                    st.warning(
                        "⚠️ The transaction probability "
                        "is above the configured fraud "
                        "detection threshold."
                    )

                else:

                    st.info(
                        "ℹ️ The transaction probability "
                        "is below the configured fraud "
                        "detection threshold."
                    )


            # ------------------------------------------------
            # API error
            # ------------------------------------------------

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.code(
                    response.text
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "🔴 Could not connect to FastAPI.\n\n"
                "Make sure FastAPI is running in another terminal:\n\n"
                "uvicorn api.main:app --reload"
            )


        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Request timed out. "
                "Please try again."
            )


        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
    <small>
    Credit Card Fraud Detection System |
    Machine Learning + FastAPI + Streamlit
    </small>
    </div>
    """,
    unsafe_allow_html=True
)