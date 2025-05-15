import streamlit as st
import pandas as pd
import requests
import openai
import io
import re
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Streamlit App Config
st.set_page_config("KoboToolbox AI Analyzer", layout="wide")
st.title("📊 KoboToolbox (https://eu.kobotoolbox.org) + AI Data Analysis")

# Sidebar: Credentials
kobo_username = st.sidebar.text_input("🔐 Kobo Username")
kobo_password = st.sidebar.text_input("🔑 Kobo Password", type="password")
openai.api_key = st.sidebar.text_input("🧠 OpenAI API Key (Optional)", type="password")
base_url = "https://eu.kobotoolbox.org"  # or change for another Kobo server

# Utility Functions
@st.cache_data(show_spinner=False)
def rename_columns(df, label_map):
    return df.rename(columns=lambda x: label_map.get(x, x))

@st.cache_data(show_spinner=False)
def map_choice_labels(df, choice_columns, choice_map):
    df_copy = df.copy()
    for col, list_name in choice_columns.items():                    
        if col in df_copy.columns and list_name in choice_map:
            df_copy[col] = df_copy[col].map(choice_map[list_name]).fillna(df_copy[col])
       
    return df_copy

@st.cache_data(show_spinner=False)
def fetch_xlsform(auth, asset_uid):
    xls_download_url = f"{base_url}/api/v2/assets/{asset_uid}.xls"
    xls_data = requests.get(xls_download_url, auth=auth).content
    return pd.read_excel(io.BytesIO(xls_data), sheet_name=None)


@st.cache_data(show_spinner=False)
def fetch_assets(auth):
    assets_url = f"{base_url}/api/v2/assets.json"
    response = requests.get(assets_url, auth=auth)
    if response.status_code == 200:
        return response.json()["results"]
    return []

@st.cache_data(show_spinner=False)
def fetch_csv(auth, form_uid):
    data_url = f"{base_url}/api/v2/assets/{form_uid}/data.json"
    resp = requests.get(data_url, auth=auth)
    if resp.status_code == 200 and len(resp.json()["results"]) > 0 :
        # Convert to DataFrame
        df = pd.DataFrame(resp.json()["results"])
        # Check for duplicates
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            print(f'Duplicate column names found: {duplicates}')
            # Remove duplicate columns, keeping the first occurrence
            df = df.loc[:, ~df.columns.duplicated(keep=False)]
        return df
    return pd.DataFrame()

# Authenticate KoboToolbox
if kobo_username and kobo_password:
    auth = (kobo_username, kobo_password)

    with st.spinner("Connecting to KoboToolbox..."):
        assets = fetch_assets(auth)
        form_choices = {f["name"]: f["uid"] for f in assets}

    if form_choices:
        if "selected_form" not in st.session_state:
            st.session_state.selected_form = list(form_choices.keys())[0]

        form_name = st.selectbox("📋 Select a Kobo Form", list(form_choices.keys()), index=list(form_choices.keys()).index(st.session_state.selected_form))
        st.session_state.selected_form = form_name
        form_uid = form_choices[form_name]

        # XLSForm Parsing
        with st.spinner("Fetching XLSForm..."):
            xls = fetch_xlsform(auth, form_uid)
            label_map, choice_map, choice_columns = {}, {}, {}
            if xls:
                st.success("XLSForm fetched successfully.")
                survey_df = xls.get("survey")
                choices_df = xls.get("choices")
                
                if survey_df is not None:
                    label_map = dict(zip(survey_df["name"], survey_df["label"]))
                    for _, row in survey_df.iterrows():
                        qtype = str(row.get("type", ""))
                        match = re.match(r"(select_one|select_multiple)\s+(\w+)", qtype)
                        if match:
                            _, list_name = match.groups()
                            choice_columns[row["name"]] = list_name
                if choices_df is not None:
                    label_col_name = "label"
                    if "label::English" in choices_df.columns:
                        label_col_name = "label::English"
                    for list_name in choices_df["list_name"].unique():
                        sub = choices_df[choices_df["list_name"] == list_name]
                        choice_map[list_name] = dict(zip(sub["name"], sub[label_col_name]))

            else:
                st.warning("Could not fetch XLSForm.")

        if st.button("📥 Load Responses"):
            df = fetch_csv(auth, form_uid)
            df.dropna(axis=1, how="all", inplace=True)

            if not df.empty:
                # if label_map:
                #     df = rename_columns(df, label_map)
                if choice_columns and choice_map:
                    df = map_choice_labels(df, choice_columns, choice_map)

                duplicates = df.columns[df.columns.duplicated()].tolist()
                if duplicates:
                    print(f'Duplicate column names found: {duplicates}')
                    # Remove duplicate columns, keeping the first occurrence
                    df = df.loc[:, ~df.columns.duplicated(keep=False)]

                st.session_state.df = df
                st.session_state.analyze_clicked = False
                st.success(f"{len(df)} responses loaded and decoded.")
                st.dataframe(df.head())

        if "df" in st.session_state:
            df = st.session_state.df

            st.subheader("📈 Descriptive Stats")
            st.dataframe(df.describe(include='all'))

            if openai.api_key:
                with st.spinner("🔍 Generating AI Insights..."):
                    prompt = f"Analyze the following dataset:\n{df.describe(include='all').to_string()}\nGive insights, patterns, and recommendations."
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )
                    ai_text = response.choices[0].message.content
                    st.markdown("### 🤖 AI Insights")
                    st.markdown(ai_text)
            else:
                st.warning("Enter your OpenAI API key to get AI insights.")

            st.subheader("📊 Auto Visualizations")
            num_cols = df.columns
            if len(num_cols):
                col = st.selectbox("Select numeric column", num_cols, key="num_col")
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                st.plotly_chart(fig)

            cat_cols = df.columns

            st.subheader("📝 Sentiment Analysis & Word Cloud")
            text_cols = df.select_dtypes(include='object').columns.tolist()
            text_col = st.selectbox("Select text column", text_cols)

            if "analyze_clicked" not in st.session_state:
                st.session_state.analyze_clicked = False

            if st.button("🔍 Analyze Text Column"):
                st.session_state.analyze_clicked = True

            if st.session_state.analyze_clicked:
                texts = df[text_col].dropna().astype(str).tolist()

                wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
                st.image(wc.to_array(), caption="Word Cloud", use_column_width=True)

                if openai.api_key:
                    prompt = f"Analyze the overall sentiment of the following responses:\n{texts[:1000]}\n\nGive a summary of tone, main themes, and emotion."
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5
                    )
                    st.markdown("### 🧠 Sentiment Summary")
                    st.markdown(response.choices[0].message.content)

            if st.button("🔄 Reset Analysis"):
                st.session_state.analyze_clicked = False

    else:
        st.error("No forms found. Check your Kobo account.")
else:
    st.info("Enter your Kobo credentials to connect.")