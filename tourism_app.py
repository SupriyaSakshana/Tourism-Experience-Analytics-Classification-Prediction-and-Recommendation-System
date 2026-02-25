import streamlit as st
import pandas as pd
import joblib

# =====================================================
# 🎡 PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
)

# =====================================================
# 🎨 CUSTOM STYLING
# =====================================================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #dbeafe, #f0f9ff);
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 10px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    color: #1e3a8a;
    margin-top: 20px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-top: 15px;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1e40af;
    color: white;
}

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #1e3a8a, #2563eb);
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 📌 SIDEBAR
# =====================================================
st.sidebar.title("🌍 Tourism Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["⭐ Rating Prediction",
     "🧭 Visit Mode Prediction",
     "🎯 Recommendation System"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Powered Tourism Dashboard")

# =====================================================
# ⚡ LOADERS
# =====================================================
@st.cache_resource
def load_rating_assets():
    encoders = joblib.load("models/encoder_rating.joblib")
    model = joblib.load("models/best_lgb_model.joblib")
    return encoders, model


@st.cache_resource
def load_visit_assets():
    encoders = joblib.load("models/encoder_visit.joblib")
    model = joblib.load("models/best_lgb_model.joblib")
    return encoders, model


@st.cache_resource
def load_content_assets():
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    knn = joblib.load("models/knn_model.joblib")
    df_lookup = pd.read_csv("DataSets/attraction_lookup.csv")

    df_lookup["combined_features"] = df_lookup[
        ["AttractionType", "CityName", "Attraction", "Rating"]
    ].astype(str).agg(" ".join, axis=1)

    tfidf_matrix = tfidf.transform(df_lookup["combined_features"])

    return tfidf, knn, df_lookup, tfidf_matrix


@st.cache_resource
def load_user_based_data():
    user_matrix = pd.read_csv("DataSets/user_attraction_matrix.csv")
    similarity_df = pd.read_csv("DataSets/attraction_similarity.csv", index_col=0)
    return user_matrix, similarity_df


# =====================================================
# 🔐 UTILITIES
# =====================================================
def safe_encode(input_df, encoders):
    for col in input_df.columns:
        if col in encoders:
            try:
                input_df[col] = encoders[col].transform(
                    input_df[col].astype(str)
                )
            except ValueError:
                st.error(f"Unknown category in column: {col}")
                return None
    return input_df


def align_features(input_df, model):
    return input_df.reindex(columns=model.feature_name_, fill_value=0)


# =====================================================
# ⭐ RATING PREDICTION
# =====================================================
if page == "⭐ Rating Prediction":

    st.markdown('<div class="main-title">⭐ Rating Prediction</div>',
                unsafe_allow_html=True)

    encoders, model = load_rating_assets()

    with st.form("rating_form"):
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("Country", encoders['Country'].classes_)
            city = st.selectbox("City", encoders['CityName'].classes_)
            attraction = st.selectbox("Attraction", encoders['Attraction'].classes_)
            attraction_type = st.selectbox("Attraction Type", encoders['AttractionType'].classes_)

        with col2:
            visit_mode = st.selectbox("Visit Mode", encoders['VisitMode'].classes_)
            year = st.selectbox("Visit Year", list(range(2013, 2031)))
            month = st.selectbox("Visit Month", list(range(1, 13)))
            user_id = st.number_input("User ID", min_value=0)

        submit = st.form_submit_button("Predict Rating")

    if submit:
        input_df = pd.DataFrame({
            'UserId': [user_id],
            'VisitYear': [year],
            'VisitMonth': [month],
            'VisitMode': [visit_mode],
            'Attraction': [attraction],
            'AttractionType': [attraction_type],
            'CityName': [city],
            'Country': [country],
        })

        input_df = safe_encode(input_df, encoders)

        if input_df is not None:
            input_df = align_features(input_df, model)
            prediction = model.predict(input_df)[0]

            st.markdown(f"""
            <div class="card">
                <h3>🌟 Predicted Rating</h3>
                <h2 style="color:#2563eb;">{prediction:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# 🧭 VISIT MODE PREDICTION
# =====================================================
elif page == "🧭 Visit Mode Prediction":

    st.markdown('<div class="main-title">🧭 Visit Mode Prediction</div>',
                unsafe_allow_html=True)

    encoders, model = load_visit_assets()

    with st.form("visit_form"):
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("Country", encoders['Country'].classes_)
            city = st.selectbox("City", encoders['CityName'].classes_)
            attraction_type = st.selectbox("Attraction Type", encoders['AttractionType'].classes_)

        with col2:
            year = st.selectbox("Visit Year", list(range(2013, 2031)))
            month = st.selectbox("Visit Month", list(range(1, 13)))
            user_id = st.number_input("User ID", min_value=0)

        submit = st.form_submit_button("Predict Visit Mode")

    if submit:
        input_df = pd.DataFrame({
            'UserId': [user_id],
            'VisitYear': [year],
            'VisitMonth': [month],
            'AttractionType': [attraction_type],
            'CityName': [city],
            'Country': [country],
        })

        input_df = safe_encode(input_df, encoders)

        if input_df is not None:
            input_df = align_features(input_df, model)
            pred = model.predict(input_df)[0]
            label = encoders['VisitMode'].inverse_transform([pred])[0]

            st.markdown(f"""
            <div class="card">
                <h3>🧭 Predicted Visit Mode</h3>
                <h2 style="color:#1e40af;">{label}</h2>
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# 🎯 RECOMMENDATION SYSTEM
# =====================================================
elif page == "🎯 Recommendation System":

    st.markdown('<div class="main-title">🎯 Recommendation System</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👥 User-Based", "📌 Content-Based"])

    # -------- USER BASED --------
    with tab1:
        user_matrix, similarity_df = load_user_based_data()
        user_id = st.selectbox("Select User",
                               sorted(user_matrix["UserId"].unique()))
        top_n = st.slider("Recommendations", 1, 10, 5)

        if st.button("Generate Recommendations"):
            matrix = user_matrix.set_index("UserId")
            user_ratings = matrix.loc[user_id]
            visited = user_ratings[user_ratings > 0].index.tolist()

            scores = {}

            for attraction in visited:
                for similar_attr, sim_score in similarity_df[attraction].items():
                    if similar_attr not in visited:
                        scores[similar_attr] = scores.get(similar_attr, 0) + sim_score

            recommendations = sorted(scores.items(),
                                     key=lambda x: x[1],
                                     reverse=True)[:top_n]

            for attr, score in recommendations:
                st.markdown(f"""
                <div class="card">
                    <b>{attr}</b><br>
                    Score: {score:.2f}
                </div>
                """, unsafe_allow_html=True)

    # -------- CONTENT BASED --------
    with tab2:
        tfidf, knn, df_lookup, tfidf_matrix = load_content_assets()
        attraction_list = sorted(df_lookup["Attraction"].unique())
        selected_attr = st.selectbox("Select Attraction", attraction_list)

        if st.button("Find Similar Attractions"):
            match = df_lookup[df_lookup["Attraction"] == selected_attr]

            if not match.empty:
                idx = match.index[0]
                n_neighbors = min(6, len(df_lookup))
                distances, indices = knn.kneighbors(
                    tfidf_matrix[idx],
                    n_neighbors=n_neighbors
                )

                for i in range(1, len(indices[0])):
                    rec = df_lookup.iloc[indices[0][i]]
                    st.markdown(f"""
                    <div class="card">
                        <b>{rec['Attraction']}</b><br>
                        📍 {rec['CityName']}<br>
                        🏷️ {rec['AttractionType']}<br>
                        ⭐ {rec['Rating']}
                    </div>
                    """, unsafe_allow_html=True)