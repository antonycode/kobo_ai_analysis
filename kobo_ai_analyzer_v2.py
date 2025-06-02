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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO
from requests.auth import HTTPBasicAuth


# Streamlit App Config
st.set_page_config("Excel, CSV, KoboToolbox, ODK, DHIS2 + AI Analyzer", layout="wide")
st.title("📊 Excel, CSV, KoboToolbox, ODK, DHIS2 + AI Data Analysis")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stActionButton {visibility: hidden;} /* Optional: hides some app buttons */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



st.sidebar.header("📥 Data Source")

data_source = st.sidebar.selectbox("Select Data Source", ["Excel", "CSV","KoboToolbox", "ODK", "DHIS2"])


# global dataframe to hold the selected data
df = None  
kobo_username = ""
kobo_password=""
base_url=""
ODK_URL = ""
API_TOKEN = ""

DHIS2_URL=""

#--------ODK -----------------------------

def get_projects():
    response = requests.get(f"{ODK_URL}/v1/projects", headers=headers)
    response.raise_for_status()
    return response.json()

def get_forms(project_id):
    response = requests.get(f"{ODK_URL}/v1/projects/{project_id}/forms", headers=headers)
    response.raise_for_status()
    return response.json()

def get_submissions(project_id, form_id):
    url = f"{ODK_URL}/v1/projects/{project_id}/forms/{form_id}.svc/Submissions?$top=1000"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["value"]

# ------DHIS2 ----------------------
@st.cache_data
def get_data_elements(auth):
    url = f"{DHIS2_URL}/api/dataElements?paging=false&fields=id,name,categoryCombo[id,name,categories[id,name,categoryOptions[id,name]]]"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json()["dataElements"]

@st.cache_data
def get_org_units(auth):
    url = f"{DHIS2_URL}/api/organisationUnits?paging=false&fields=id,name,path"
    r = requests.get(url, auth=auth)
    r.raise_for_status()
    return r.json()["organisationUnits"]

# --- 1. Excel Upload ---
if data_source == "Excel":
    excel_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if excel_file:
        sheet_names = pd.ExcelFile(excel_file).sheet_names
        selected_sheet = st.sidebar.selectbox("Select Sheet", sheet_names)
        df = pd.read_excel(excel_file, sheet_name=selected_sheet)
        st.session_state.df = df
        st.session_state.analyze_clicked = False
        st.success(f"{len(df)} responses loaded and decoded.")
        st.dataframe(df.head())
        st.success("✅ Excel data loaded")
# --- 1a. CSV Upload ---
if data_source == "CSV":
    csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.session_state.df = df
        st.session_state.analyze_clicked = False
        st.success(f"{len(df)} responses loaded and decoded.")
        st.dataframe(df.head())
        st.success("✅ CSV data loaded")

# --- 2. KoboToolbox (ODK) ---
elif data_source == "KoboToolbox":

    # Sidebar: Credentials
    base_url = st.sidebar.selectbox(
        "Select your KoboToolbox server",
        options=[
            "https://eu.kobotoolbox.org",
            "https://kf.kobotoolbox.org"
        ]
    )
    kobo_username = st.sidebar.text_input("🔐 Kobo Username")
    kobo_password = st.sidebar.text_input("🔑 Kobo Password", type="password")

# --- 3. DHIS2 ---
elif data_source == "DHIS2":
    st.markdown("### 📦 Coming Soon.....................")
    # st.sidebar.info("DHIS2 credentials & URL required")
    DHIS2_URL = st.sidebar.text_input("🌍 DHIS2 URL (e.g. https://play.im.dhis2.org/stable-2-41-4)")
    dhis_username = st.sidebar.text_input("👤 Username")
    dhis_password = st.sidebar.text_input("🔐 Password", type="password")
    pe = st.sidebar.text_input("Period (e.g. 202301;202302)", value="LAST_12_MONTHS")

    if st.sidebar.button("🔄 Load DHIS2 Data"):
        if DHIS2_URL and dhis_username and dhis_password  and pe:
            try:
                auth = (dhis_username,dhis_password)
                data_elements = get_data_elements(auth)
                de_options = {f"{de['name']} ({de['id']})": de for de in data_elements}

                selected_de_name = st.selectbox("🔢 Select Data Element", list(de_options.keys()))
                selected_de = de_options[selected_de_name]

                st.markdown("### 📦 Category Option Combos")

                if "categoryCombo" in selected_de:
                    categories = selected_de["categoryCombo"].get("categories", [])
                    for cat in categories:
                        options = [opt["name"] for opt in cat.get("categoryOptions", [])]
                        st.selectbox(f"{cat['name']}", options)

                org_units = get_org_units(auth)
                ou_options = {ou["name"]: ou["id"] for ou in org_units}
                selected_ou = st.selectbox("📍 Select Organisation Unit", list(ou_options.keys()))

                st.markdown("### 📦 Coming Soon.....................")

                #url = f"{dhis_url}/api/analytics.json?dimension=dx:{dx}&dimension=ou:{ou}&dimension=pe:{pe}&displayProperty=NAME"
                # Example: fetch data values for this DE, OU and Period
                # url = f"{DHIS2_URL}/api/dataValueSets?dataElement={selected_de['id']}&orgUnit={ou_options[selected_ou]}&period=202401"
                # r = requests.get(url, auth=(dhis_username, dhis_password))
                # if r.status_code == 200:
                #     data = r.json()
                #     df = pd.DataFrame(data.get("dataValues", []))
                #     if df.empty:
                #         st.info("ℹ️ No data values found for this selection.")
                # else:
                #     st.dataframe(df)
                # st.session_state.df = df
                # st.session_state.analyze_clicked = False
                # st.success(f"{len(df)} rs loaded and decoded.")
                # st.dataframe(df.head())
                # st.success(f"✅ {len(df)} DHIS2 records loaded")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("Please fill in all DHIS2 fields.")

# --- 4. (ODK) ---
elif data_source == "ODK":

    # Sidebar: Credentials
    ODK_URL = st.sidebar.text_input("🌍 ODK Server URL (e.g. https://your-odk-central.org)")
    API_TOKEN = odk_username = st.sidebar.text_input("🔐 ODK API Token")
    #odk_username = st.sidebar.text_input("🔐 ODK Username")
    #odk_password = st.sidebar.text_input("🔑 ODK Password", type="password")

    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    # --- Streamlit UI ---
    if ODK_URL and API_TOKEN:
        # Get all forms from all projects
        project_form_pairs = []
        projects = get_projects()
        for project in projects:
            forms = get_forms(project["id"])
            for form in forms:
                label = f"{project['name']} → {form['name']} ({form['xmlFormId']})"
                project_form_pairs.append((label, project["id"], form["xmlFormId"]))

        # Selectbox: choose form
        form_label = st.selectbox("Select a form to download submissions", [p[0] for p in project_form_pairs])

        # Get the selected project/form ID
        selected = next(p for p in project_form_pairs if p[0] == form_label)
        project_id, form_id = selected[1], selected[2]

        # Download submissions
        if st.button("Download Submissions"):
            with st.spinner("Fetching submissions..."):
                try:
                    submissions = get_submissions(project_id, form_id)
                    if submissions:
                        df = pd.DataFrame(submissions)
                        st.session_state.df = df
                        st.session_state.analyze_clicked = False
                        st.success(f"{len(df)} responses loaded and decoded.")
                        st.dataframe(df.head())
                        st.success(f"✅ {len(df)} DHIS2 records loaded")
                        st.dataframe(df)
                    else:
                        st.info("No submissions found for this form.")
                except Exception as e:
                    st.error(f"Error fetching submissions: {e}")


openai.api_key = st.sidebar.text_input("🧠 OpenAI API Key (Optional)", type="password")

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
def fetch_kobo_csv(auth, form_uid):
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

@st.cache_data(show_spinner=False)
def fetch_kobo_csv(auth, form_uid):
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

            # 
            df = fetch_kobo_csv(auth, form_uid)
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

    else:
        st.error("No forms found. Check your Kobo account.")
# else:
#     st.info("Enter your Kobo credentials to connect.")

#This now applies to all the Data Sources >>> Since we have a DF

if "df" in st.session_state:
    df = st.session_state.df

    # Download responses
    if not df.empty:
        st.subheader("📥 Download Responses")

        # Choose format
        download_format = st.radio("Choose download format:", ("CSV", "Excel (.xlsx)"))

        # Generate download link based on selection
        if download_format == "CSV":
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="responses.csv",
                mime="text/csv"
            )

        elif download_format == "Excel (.xlsx)":
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Responses")
            output.seek(0)
            st.download_button(
                label="⬇️ Download Excel",
                data=output,
                file_name="responses.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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

    num_cols = df.columns
    
    if len(num_cols):
        all_cols = df.columns.tolist()
        st.subheader("📈 Custom Chart Builder")

        # Chart type selection
        chart_type = st.selectbox("📊 Select chart type", ["Scatter", "Line", "Bar", "Pie"])

        # Shared inputs for title and axis labels
        custom_title = st.text_input("📌 Chart Title", value="My Chart")
        selected_color = st.color_picker("🎨 Pick chart color", value="#1f77b4")  # default Plotly blue

        use_secondary = st.checkbox("➕ Add secondary Y-axis")
        

        # Create figure with secondary y-axis if selected
        fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

        if chart_type in ["Scatter", "Line", "Bar"]:

            x_col = st.selectbox("🧭 X-axis", all_cols)
            y_col = st.selectbox("📏 Y-axis", [col for col in all_cols if col != x_col])

            custom_x_label = st.text_input("🖋 X-axis Label", value=x_col)
            custom_y_label = st.text_input("🖋 Y-axis Label", value=y_col)

            if use_secondary:
                y2_col = st.selectbox("📐 Secondary Y-axis", [col for col in all_cols if col not in [x_col, y_col]])
                custom_y2_label = st.text_input("🖋 Secondary Y-axis Label", value=y2_col)
                selected_color2 = st.color_picker("🎨 Secondary Y-axis Color", value="#ff7f0e")

            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)
                fig.update_traces(marker=dict(color=selected_color))
            elif chart_type == "Line":
                # Add secondary trace
                if use_secondary:
                    fig.add_trace(
                        go.Scatter(x=df[x_col], y=df[y2_col], name=y2_col, line=dict(color=selected_color2)),
                        secondary_y=True
                    )      
                else:
                    fig.add_trace(
                        go.Scatter(x=df[x_col], y=df[y_col], name=y_col, line=dict(color=selected_color)),
                        secondary_y=False
                    )
            elif chart_type == "Bar":
                if use_secondary:
                    fig.add_trace(
                            go.Bar(x=df[x_col], y=df[y2_col], name=y2_col, marker_color=selected_color2),
                            secondary_y=True
                        )
                else:
                    fig.add_trace(
                        go.Bar(x=df[x_col], y=df[y_col], name=y_col, marker_color=selected_color),
                        secondary_y=False
                    )

            # Apply custom titles and labels
            # Layout settings
            fig.update_layout(
                title_text=custom_title,
                legend_title="Legend",
                xaxis_title=custom_x_label
            )
            fig.update_yaxes(title_text=custom_y_label, secondary_y=False)
            if use_secondary:
                fig.update_yaxes(title_text=custom_y2_label, secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

        # Pie chart
        elif chart_type == "Pie":
            use_count = st.checkbox("🔢 Use count of each unique value (no need for numeric values)")

            if use_count:
                pie_label = st.selectbox("🧩 Pie slices (labels)", all_cols)
                custom_pie_title = st.text_input("📌 Pie Chart Title", value=f"Count of {pie_label}")
                
                # Count unique values
                pie_data = df[pie_label].value_counts().reset_index()
                pie_data.columns = [pie_label, "Count"]

                fig = px.pie(pie_data, names=pie_label, values="Count", color_discrete_sequence=[selected_color])
                fig.update_layout(title=custom_pie_title)

            else:
                pie_label = st.selectbox("🧩 Pie slices (labels)", all_cols)
                pie_value = st.selectbox("🔢 Pie values", all_cols)
                custom_pie_title = st.text_input("📌 Pie Chart Title", value=f"{pie_value} by {pie_label}")
                
                fig = px.pie(df, names=pie_label, values=pie_value, color_discrete_sequence=[selected_color])
                fig.update_layout(title=custom_pie_title)

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺️ Location Map (if latitude/longitude available)")
    # Detect GPS fields in single column
    gps_cols = [ col for col in df.columns if df[col].dropna().apply(lambda x: len(str(x).split()) == 4).all() ]

    if gps_cols:
        gps_col = st.selectbox("Select GPS field", gps_cols, help="Field containing space-separated lat/lon")
        #df[['lat', 'lon']] = df[gps_col].dropna().apply( lambda x: pd.Series(str(x).split()[:2]), axis=1).astype(float)
        df[['lat', 'lon']] = df[gps_col].dropna().apply(lambda x: pd.Series(str(x).split()[:2])).astype(float)

        st.map(df[['lat', 'lon']].dropna())

    else:
        st.info("No GPS column with space-separated coordinates found.")

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
        st.image(wc.to_array(), caption="Word Cloud", use_container_width=True)

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

