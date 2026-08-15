import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "laptop_price_pipeline.pkl"
)

OPTIONS_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "input_options.pkl"
)

st.markdown(
    """
<style>
/* ---------------------------------------------------------
   GOOGLE FONT
--------------------------------------------------------- */
@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */
html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(14, 165, 233, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 90%,
            rgba(20, 184, 166, 0.08),
            transparent 30%
        ),
}

.main .block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: #091525;
    border-right: 1px solid rgba(148, 163, 184, 0.15);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

section[data-testid="stSidebar"] p {
    color: #b8c5d6;
}

/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */
.hero-container {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.20);
    border-radius: 26px;
    padding: 38px 40px;
    margin-bottom: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(13, 31, 53, 0.98),
            rgba(9, 45, 59, 0.94)
        );
    box-shadow:
        0 25px 60px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.hero-container::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -80px;
    top: -130px;
    border-radius: 50%;
    background: rgba(34, 211, 238, 0.10);
    filter: blur(4px);
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    margin-bottom: 16px;
    border: 1px solid rgba(34, 211, 238, 0.32);
    border-radius: 999px;
    background: rgba(34, 211, 238, 0.08);
    color: #67e8f9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

.hero-title {
    color: #f8fafc;
    font-size: clamp(34px, 5vw, 54px);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -1.5px;
    margin: 0 0 14px 0;
}

.hero-title-highlight {
    color: #22d3ee;
}

.hero-description {
    max-width: 780px;
    color: #b9c7d8;
    font-size: 16px;
    line-height: 1.75;
    margin: 0;
}

/* ---------------------------------------------------------
   SECTION HEADINGS
--------------------------------------------------------- */
.section-heading {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 750;
    margin: 18px 0 5px 0;
}

.section-description {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 18px;
}

/* ---------------------------------------------------------
   METRICS
--------------------------------------------------------- */
div[data-testid="stMetric"] {
    min-height: 118px;
    padding: 20px 22px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    background: rgba(12, 27, 45, 0.86);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
}

div[data-testid="stMetricLabel"] {
    color: #91a4ba;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 750;
}

/* ---------------------------------------------------------
   FORM AND INPUTS
--------------------------------------------------------- */
div[data-testid="stForm"] {
    padding: 28px;
    border: 1px solid rgba(148, 163, 184, 0.17);
    border-radius: 22px;
    background: rgba(10, 24, 41, 0.90);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
}

label,
div[data-testid="stWidgetLabel"] p {
    color: #dbe7f4 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    color: #eaf2fb;
    background: #0d1d30;
    border-color: rgba(148, 163, 184, 0.22);
    border-radius: 11px;
}

div[data-baseweb="select"] > div:hover,
div[data-testid="stNumberInput"] input:hover {
    border-color: rgba(34, 211, 238, 0.60);
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #22d3ee;
    box-shadow: 0 0 0 1px #22d3ee;
}

/* ---------------------------------------------------------
   BUTTON
--------------------------------------------------------- */
div[data-testid="stFormSubmitButton"] button,
.stButton > button {
    width: 100%;
    min-height: 52px;
    margin-top: 10px;
    border: none;
    border-radius: 13px;
    color: #03111b;
    background: linear-gradient(
        90deg,
    );
    font-size: 16px;
    font-weight: 800;
    transition:
        transform 0.20s ease,
        box-shadow 0.20s ease;
}

div[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
    color: #03111b;
    transform: translateY(-2px);
    box-shadow: 0 13px 28px rgba(34, 211, 238, 0.23);
}

div[data-testid="stFormSubmitButton"] button:active,
.stButton > button:active {
    transform: translateY(0);
}

/* ---------------------------------------------------------
   RESULT CARD
--------------------------------------------------------- */
.result-container {
    position: relative;
    overflow: hidden;
    margin-top: 28px;
    padding: 35px 30px;
    border: 1px solid rgba(45, 212, 191, 0.30);
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(8, 47, 73, 0.98),
            rgba(13, 75, 74, 0.96)
        );
    text-align: center;
    box-shadow:
        0 22px 55px rgba(0, 0, 0, 0.32),
        0 0 35px rgba(45, 212, 191, 0.08);
}

.result-container::before {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    left: -90px;
    bottom: -130px;
    border-radius: 50%;
    background: rgba(34, 211, 238, 0.10);
}

.result-icon {
    font-size: 38px;
    margin-bottom: 8px;
}

.result-label {
    color: #b6e8e5;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.result-price {
    margin: 8px 0;
    color: #ffffff;
    font-size: clamp(38px, 6vw, 58px);
    font-weight: 850;
    letter-spacing: -1.5px;
}

.result-range {
    color: #c8f7f1;
    font-size: 14px;
    font-weight: 600;
}

.result-model {
    display: inline-block;
    margin-top: 15px;
    padding: 7px 13px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 999px;
    color: #d8f7f4;
    background: rgba(255, 255, 255, 0.07);
    font-size: 12px;
}

/* ---------------------------------------------------------
   SUMMARY CARDS
--------------------------------------------------------- */
.summary-card {
    height: 100%;
    min-height: 112px;
    padding: 18px 20px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 17px;
    background: rgba(11, 26, 44, 0.90);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
}

.summary-label {
    color: #8ea2b8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.summary-value {
    margin-top: 9px;
    color: #f8fafc;
    font-size: 20px;
    font-weight: 750;
}

/* ---------------------------------------------------------
   INFORMATION BOX
--------------------------------------------------------- */
.info-box {
    margin-top: 20px;
    padding: 17px 19px;
    border-left: 4px solid #22d3ee;
    border-radius: 11px;
    color: #b9c8d8;
    background: rgba(14, 165, 233, 0.07);
    font-size: 13px;
    line-height: 1.65;
}

/* ---------------------------------------------------------
   DATAFRAME
--------------------------------------------------------- */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    overflow: hidden;
}

/* ---------------------------------------------------------
   DIVIDER
--------------------------------------------------------- */
hr {
    border-color: rgba(148, 163, 184, 0.14);
}

/* ---------------------------------------------------------
   HIDE STREAMLIT ELEMENTS
--------------------------------------------------------- */
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 0.8rem 3rem 0.8rem;
    }

    .hero-container {
        padding: 27px 22px;
        border-radius: 20px;
    }

    .hero-title {
        font-size: 35px;
    }

    div[data-testid="stForm"] {
        padding: 20px 15px;
    }

    .result-container {
        padding: 29px 15px;
    }
}
</style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_prediction_pipeline():
    """Load complete preprocessing and prediction pipeline."""

    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Pipeline file not found:\n{PIPELINE_PATH}"
        )

    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_input_options():
    """Load dropdown options created from training dataset."""

    if not os.path.exists(OPTIONS_PATH):
        raise FileNotFoundError(
            f"Input options file not found:\n{OPTIONS_PATH}"
        )

    return joblib.load(OPTIONS_PATH)

try:
    prediction_pipeline = load_prediction_pipeline()
    input_options = load_input_options()

except Exception as error:
    st.error("Model files could not be loaded.")
    st.code(str(error))

    st.info(
        "Make sure the `saved_models` folder contains "
        "`laptop_price_pipeline.pkl` aur `input_options.pkl` "
        "files."
    )

    st.stop()

def get_clean_options(key, fallback):
    """Return clean and sorted options for select boxes."""

    values = input_options.get(key, fallback)

    cleaned_values = [
        value
        for value in values
        if pd.notna(value)
    ]

    try:
        return sorted(cleaned_values)

    except TypeError:
        return cleaned_values

def find_default_index(options, preferred_value):
    """Safely return the index of a preferred default value."""

    if preferred_value in options:
        return options.index(preferred_value)

    return 0

def calculate_ppi(
    horizontal_resolution,
    vertical_resolution,
    screen_inches
):
    """Calculate screen pixels per inch."""

    diagonal_pixels = (
        horizontal_resolution ** 2
        + vertical_resolution ** 2
    ) ** 0.5

    return diagonal_pixels / screen_inches

def create_summary_card(label, value):
    """Return a safe HTML summary card."""

    return (
        '<div class="summary-card">'
        f'<div class="summary-label">{label}</div>'
        f'<div class="summary-value">{value}</div>'
        '</div>'
    )

company_options = get_clean_options(
    "Company",
    ["Acer", "Apple", "Asus", "Dell", "HP", "Lenovo"]
)

type_options = get_clean_options(
    "TypeName",
    [
        "Notebook",
        "Ultrabook",
        "Gaming",
        "2 in 1 Convertible",
        "Workstation",
        "Netbook"
    ]
)

cpu_options = get_clean_options(
    "CPU_Brand",
    [
        "Intel Core i3",
        "Intel Core i5",
        "Intel Core i7",
        "Other Intel Processor",
        "AMD Processor"
    ]
)

gpu_options = get_clean_options(
    "GPU_Brand",
    ["AMD", "Intel", "Nvidia"]
)

os_options = get_clean_options(
    "OS",
    ["Linux", "Mac", "Other", "Windows"]
)

ram_options = get_clean_options(
    "Ram_GB",
    [2, 4, 8, 16, 32, 64]
)

ssd_options = get_clean_options(
    "SSD_GB",
    [0, 128, 256, 512, 1024]
)

hdd_options = get_clean_options(
    "HDD_GB",
    [0, 500, 1000, 2000]
)

flash_options = get_clean_options(
    "Flash_GB",
    [0, 16, 32, 64, 128, 256, 512]
)

hybrid_options = get_clean_options(
    "Hybrid_GB",
    [0, 500, 1000, 2000]
)

with st.sidebar:
    st.markdown("##  Laptop AI")

    st.caption("Machine Learning Price Prediction")

    st.divider()

    st.markdown("### Model details")

    st.metric(
        label="Final model",
        value="Random Forest"
    )

    st.metric(
        label="Test R² score",
        value="79.01%"
    )

    st.metric(
        label="Mean absolute error",
        value="₹11,007"
    )

    st.divider()

    st.markdown("### Project workflow")

    st.markdown(
        """
        1. Dataset cleaning  
        2. Feature engineering  
        3. One-hot encoding  
        4. Model comparison  
        5. Random Forest selection  
        6. Price prediction
        """
    )

    st.divider()

    st.caption(
        "Predicted value is an estimate. Actual price may differ "
        "according to market condition, generation and availability."
    )

st.markdown(
    """
<div class="hero-container">
    <div class="hero-badge">Machine Learning Project</div>
    <h1 class="hero-title">
        Find the right price for your
        <span class="hero-title-highlight">next laptop.</span>
    </h1>
    <p class="hero-description">
        Enter your laptop specifications and calculate an estimated price
        using the trained Random Forest regression model.
        The model analyses important features such as company, processor,
        RAM, storage, display, and GPU.
    </p>
</div>
    """,
    unsafe_allow_html=True
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

with metric_col_1:
    st.metric(
        label="Algorithm",
        value="Random Forest"
    )

with metric_col_2:
    st.metric(
        label="R² Score",
        value="0.7901"
    )

with metric_col_3:
    st.metric(
        label="Average Error",
        value="₹11,007"
    )

with metric_col_4:
    st.metric(
        label="Project Type",
        value="Regression"
    )

st.markdown(
    '<div class="section-heading">Configure your laptop</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section-description">
    Carefully select the laptop specifications for a more accurate prediction.
</div>
    """,
    unsafe_allow_html=True
)

with st.form(
    key="laptop_prediction_form",
    clear_on_submit=False
):
    left_column, middle_column, right_column = st.columns(3)

    with left_column:
        st.markdown("#### General details")

        company = st.selectbox(
            label="Laptop company",
            options=company_options,
            index=find_default_index(
                company_options,
                "Dell"
            )
        )

        type_name = st.selectbox(
            label="Laptop category",
            options=type_options,
            index=find_default_index(
                type_options,
                "Notebook"
            )
        )

        os_name = st.selectbox(
            label="Operating system",
            options=os_options,
            index=find_default_index(
                os_options,
                "Windows"
            )
        )

        weight_kg = st.number_input(
            label="Laptop weight (kg)",
            min_value=0.50,
            max_value=6.00,
            value=1.80,
            step=0.05,
            format="%.2f"
        )

    with middle_column:
        st.markdown("#### Performance")

        cpu_brand = st.selectbox(
            label="Processor category",
            options=cpu_options,
            index=find_default_index(
                cpu_options,
                "Intel Core i5"
            )
        )

        gpu_brand = st.selectbox(
            label="Graphics brand",
            options=gpu_options,
            index=find_default_index(
                gpu_options,
                "Intel"
            )
        )

        ram_gb = st.selectbox(
            label="RAM capacity (GB)",
            options=ram_options,
            index=find_default_index(
                ram_options,
                8
            )
        )

        ssd_gb = st.selectbox(
            label="SSD storage (GB)",
            options=ssd_options,
            index=find_default_index(
                ssd_options,
                256
            )
        )

        hdd_gb = st.selectbox(
            label="HDD storage (GB)",
            options=hdd_options,
            index=find_default_index(
                hdd_options,
                0
            )
        )

    with right_column:
        st.markdown("#### Display and extras")

        inches = st.number_input(
            label="Screen size (inches)",
            min_value=10.0,
            max_value=20.0,
            value=15.6,
            step=0.1,
            format="%.1f"
        )

        resolution = st.selectbox(
            label="Screen resolution",
            options=[
                "1366 × 768",
                "1600 × 900",
                "1920 × 1080",
                "2256 × 1504",
                "2560 × 1440",
                "2560 × 1600",
                "2880 × 1800",
                "3200 × 1800",
                "3840 × 2160"
            ],
            index=2
        )

        touchscreen_label = st.selectbox(
            label="Touchscreen display",
            options=["No", "Yes"],
            index=0
        )

        ips_label = st.selectbox(
            label="IPS panel",
            options=["No", "Yes"],
            index=1
        )

        flash_gb = st.selectbox(
            label="Flash storage (GB)",
            options=flash_options,
            index=find_default_index(
                flash_options,
                0
            )
        )

        hybrid_gb = st.selectbox(
            label="Hybrid storage (GB)",
            options=hybrid_options,
            index=find_default_index(
                hybrid_options,
                0
            )
        )

    st.markdown("")

    submitted = st.form_submit_button(
        label="Calculate estimated price",
        use_container_width=True
    )

if submitted:
    try:
        resolution_values = resolution.split("×")

        x_resolution = int(
            resolution_values[0].strip()
        )

        y_resolution = int(
            resolution_values[1].strip()
        )

        ppi = calculate_ppi(
            horizontal_resolution=x_resolution,
            vertical_resolution=y_resolution,
            screen_inches=float(inches)
        )

        touchscreen = (
            1
            if touchscreen_label == "Yes"
            else 0
        )

        ips = (
            1
            if ips_label == "Yes"
            else 0
        )

        input_data = pd.DataFrame(
            {
                "Company": [company],
                "TypeName": [type_name],
                "Inches": [float(inches)],
                "Ram_GB": [int(ram_gb)],
                "Weight_kg": [float(weight_kg)],
                "Touchscreen": [int(touchscreen)],
                "IPS": [int(ips)],
                "PPI": [float(ppi)],
                "CPU_Brand": [cpu_brand],
                "SSD_GB": [int(ssd_gb)],
                "HDD_GB": [int(hdd_gb)],
                "Flash_GB": [int(flash_gb)],
                "Hybrid_GB": [int(hybrid_gb)],
                "GPU_Brand": [gpu_brand],
                "OS": [os_name]
            }
        )

        raw_prediction = prediction_pipeline.predict(
            input_data
        )[0]

        predicted_price = max(
            float(raw_prediction),
            0.0
        )

        estimated_error = 11006.72

        lower_price = max(
            predicted_price - estimated_error,
            0
        )

        upper_price = (
            predicted_price + estimated_error
        )

        result_html = (
            '<div class="result-container">'
            '<div class="result-icon">💻</div>'
            '<div class="result-label">'
            'Estimated Laptop Price'
            '</div>'
            '<div class="result-price">'
            f'₹{predicted_price:,.0f}'
            '</div>'
            '<div class="result-range">'
            f'Expected range: ₹{lower_price:,.0f} – '
            f'₹{upper_price:,.0f}'
            '</div>'
            '<div class="result-model">'
            'Prediction generated using Random Forest'
            '</div>'
            '</div>'
        )

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-heading">'
            'Selected configuration'
            '</div>',
            unsafe_allow_html=True
        )

        summary_col_1, summary_col_2, summary_col_3, summary_col_4 = (
            st.columns(4)
        )

        with summary_col_1:
            st.markdown(
                create_summary_card(
                    "Company",
                    company
                ),
                unsafe_allow_html=True
            )

        with summary_col_2:
            st.markdown(
                create_summary_card(
                    "Processor",
                    cpu_brand
                ),
                unsafe_allow_html=True
            )

        with summary_col_3:
            st.markdown(
                create_summary_card(
                    "Memory",
                    f"{ram_gb} GB RAM"
                ),
                unsafe_allow_html=True
            )

        with summary_col_4:
            storage_text = (
                f"{ssd_gb} GB SSD"
                if int(ssd_gb) > 0
                else f"{hdd_gb} GB HDD"
            )

            st.markdown(
                create_summary_card(
                    "Primary storage",
                    storage_text
                ),
                unsafe_allow_html=True
            )

        second_summary_1, second_summary_2, second_summary_3, second_summary_4 = (
            st.columns(4)
        )

        with second_summary_1:
            st.markdown(
                create_summary_card(
                    "Screen",
                    f"{inches} inch"
                ),
                unsafe_allow_html=True
            )

        with second_summary_2:
            st.markdown(
                create_summary_card(
                    "Resolution",
                    resolution
                ),
                unsafe_allow_html=True
            )

        with second_summary_3:
            st.markdown(
                create_summary_card(
                    "Graphics",
                    gpu_brand
                ),
                unsafe_allow_html=True
            )

        with second_summary_4:
            st.markdown(
                create_summary_card(
                    "Operating system",
                    os_name
                ),
                unsafe_allow_html=True
            )

        with st.expander(
            "View complete prediction details"
        ):
            detail_table = pd.DataFrame(
                {
                    "Specification": [
                        "Company",
                        "Laptop Type",
                        "Processor",
                        "Graphics",
                        "Operating System",
                        "RAM",
                        "SSD",
                        "HDD",
                        "Flash Storage",
                        "Hybrid Storage",
                        "Screen Size",
                        "Resolution",
                        "Touchscreen",
                        "IPS Panel",
                        "Weight",
                        "PPI",
                        "Predicted Price",
                        "Prediction Time"
                    ],
                    "Selected Value": [
                        company,
                        type_name,
                        cpu_brand,
                        gpu_brand,
                        os_name,
                        f"{ram_gb} GB",
                        f"{ssd_gb} GB",
                        f"{hdd_gb} GB",
                        f"{flash_gb} GB",
                        f"{hybrid_gb} GB",
                        f"{inches} inches",
                        resolution,
                        touchscreen_label,
                        ips_label,
                        f"{weight_kg:.2f} kg",
                        f"{ppi:.2f}",
                        f"₹{predicted_price:,.2f}",
                        datetime.now().strftime(
                            "%d %B %Y, %I:%M %p"
                        )
                    ]
                }
            )

            st.dataframe(
                detail_table,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            """
<div class="info-box">
    <strong>Important:</strong>
    This price is an estimate generated by the machine learning model.
    The actual laptop price may vary depending on the model generation,
    processor generation, warranty, location, seller, and current offers.
</div>
            """,
            unsafe_allow_html=True
        )

    except Exception as error:
        st.error(
            "Prediction could not be generated. "
            "Please check the error shown below."
        )

        st.code(str(error))