import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AeroSegment | Grocery ML Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# API CONFIG
# ============================================================
import os
API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"],
    .stMarkdown, p, div, label,
    span, button {
        font-family: 'Outfit', sans-serif !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background:
        radial-gradient(
            circle at 10% 20%,
            rgb(18, 20, 32) 0%,
            rgb(10, 11, 20) 100%
        ) !important;

        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(12, 14, 24, 0.98) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .stButton > button {
        background:
        linear-gradient(
            135deg,
            #6366f1 0%,
            #a855f7 100%
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;

        transition: all 0.25s ease !important;

        box-shadow:
        0 4px 15px
        rgba(99,102,241,0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;

        box-shadow:
        0 8px 25px
        rgba(168,85,247,0.45) !important;
    }

    div[data-testid="stMetric"] {
        background:
        rgba(255,255,255,0.025);

        border:
        1px solid rgba(255,255,255,0.06);

        border-radius: 15px;

        padding: 15px;

        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);

        border-color:
        rgba(168,85,247,0.35);
    }

    .hero-card {
        padding: 35px;
        border-radius: 22px;

        background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.16),
            rgba(168,85,247,0.10)
        );

        border:
        1px solid rgba(168,85,247,0.20);

        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;

        background:
        linear-gradient(
            90deg,
            #ffffff,
            #c4b5fd,
            #67e8f9
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 17px;
    }

    .result-card {
        padding: 30px;
        border-radius: 20px;

        background:
        linear-gradient(
            135deg,
            rgba(56,189,248,0.10),
            rgba(168,85,247,0.10)
        );

        border:
        1px solid rgba(255,255,255,0.08);

        margin-top: 20px;
    }

    .cluster-number {
        font-size: 45px;
        font-weight: 800;
        color: #ffffff;
    }

    .cluster-name {
        font-size: 25px;
        font-weight: 700;
        color: #c4b5fd;
    }

    .status-connected {
        padding: 10px 15px;
        border-radius: 10px;

        background:
        rgba(34,197,94,0.12);

        color: #4ade80;

        text-align: center;
        font-weight: 600;
    }

    .status-offline {
        padding: 10px 15px;
        border-radius: 10px;

        background:
        rgba(239,68,68,0.12);

        color: #f87171;

        text-align: center;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API FUNCTIONS
# ============================================================

@st.cache_data(ttl=30)
def check_backend():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:

        return False


@st.cache_data(ttl=300)
def get_model_info():

    try:

        response = requests.get(
            f"{API_URL}/model-info",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:

        return None


@st.cache_data(ttl=300)
def get_analytics():

    try:

        response = requests.get(
            f"{API_URL}/analytics-data",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:

        return None


def predict_product(payload):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15
        )

        return response

    except requests.exceptions.RequestException as e:

        return None


# ============================================================
# LOAD BACKEND
# ============================================================

backend_online = check_backend()

model_metadata = (
    get_model_info()
    if backend_online
    else None
)

analytics_data = (
    get_analytics()
    if backend_online
    else None
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px 0 25px 0;
        ">

        <div style="
            font-size:26px;
            font-weight:800;
        ">
        🧬 AeroSegment
        </div>

        <div style="
            color:#64748b;
            font-size:13px;
            margin-top:5px;
        ">
        AI Grocery Segmentation
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    menu = st.selectbox(
        "Menu Navigation",
        [
            "🔮 Product Analyzer",
            "📊 Segment Analytics",
            "🧬 Explore Clusters",
            "📖 About the Model"
        ]
    )

    st.write("")

    if backend_online:

        st.markdown(
            """
            <div class="status-connected">
            🟢 Backend Connected
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-offline">
            🔴 Backend Offline
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    if st.button(
        "🔄 Refresh Connection",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


# ============================================================
# CONNECTION GUARD
# ============================================================

if not backend_online:

    st.markdown(
        """
        <div class="hero-card">

        <div class="hero-title">
        🔌 Service Connection Offline
        </div>

        <div class="hero-subtitle">
        The FastAPI backend is currently unavailable.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.error(
        "Could not connect to FastAPI."
    )

    st.info(
        "Start the backend from the project root:"
    )

    st.code(
        "python -m uvicorn backend.main:app --reload",
        language="bash"
    )

    st.write(
        "Backend URL:"
    )

    st.code(
        API_URL
    )

    st.stop()


# ============================================================
# SAFETY CHECK
# ============================================================

if model_metadata is None:

    st.error(
        "Backend is running but /model-info failed."
    )

    st.stop()


# ============================================================
# EXTRACT METADATA
# ============================================================

n_clusters = model_metadata.get(
    "number_of_clusters",
    0
)

features = model_metadata.get(
    "features",
    []
)

dataset_info = model_metadata.get(
    "dataset",
    {}
)

overall_stats = dataset_info.get(
    "statistics",
    {}
)

clusters_db = model_metadata.get(
    "clusters",
    {}
)

evaluation = model_metadata.get(
    "evaluation",
    {}
)


# ============================================================
# HERO
# ============================================================

if menu == "🔮 Product Analyzer":

    st.markdown(
        """
        <div class="hero-card">

        <div class="hero-title">
        🛒 AI Grocery Product Segmentation
        </div>

        <div class="hero-subtitle">
        Intelligent grocery product segmentation
        powered by K-Means Machine Learning.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PRODUCT ANALYZER
# ============================================================

if menu == "🔮 Product Analyzer":

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Total Products",
        f"{int(overall_stats.get('total_products', 0)):,}"
    )

    k2.metric(
        "Clusters",
        n_clusters
    )

    k3.metric(
        "Average Price",
        f"₹{overall_stats.get('avg_price', 0):,.2f}"
    )

    k4.metric(
        "Average Rating",
        f"{overall_stats.get('avg_rating', 0):.2f} ⭐"
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT STATE
    # --------------------------------------------------------

    if "input_price" not in st.session_state:

        st.session_state.input_price = 500.0
        st.session_state.input_discount = 50.0
        st.session_state.input_rating = 4.5
        st.session_state.input_reviews = 1000
        st.session_state.input_title_length = 30
        st.session_state.input_feature_length = 100
        st.session_state.input_description_length = 300


    def set_template(
        price,
        discount,
        rating,
        reviews,
        title_length,
        feature_length,
        description_length
    ):

        st.session_state.input_price = price
        st.session_state.input_discount = discount
        st.session_state.input_rating = rating
        st.session_state.input_reviews = reviews
        st.session_state.input_title_length = title_length
        st.session_state.input_feature_length = feature_length
        st.session_state.input_description_length = description_length

        st.rerun()


    # --------------------------------------------------------
    # SAMPLE PRODUCTS
    # --------------------------------------------------------

    st.subheader(
        "⚡ Quick Product Templates"
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        if st.button(
            "💰 Budget Product",
            use_container_width=True
        ):

            set_template(
                150,
                20,
                4.1,
                100,
                35,
                70,
                250
            )

    with t2:

        if st.button(
            "💎 Premium Product",
            use_container_width=True
        ):

            set_template(
                1200,
                50,
                4.8,
                1500,
                70,
                180,
                1200
            )

    with t3:

        if st.button(
            "🏷️ Discount Product",
            use_container_width=True
        ):

            set_template(
                300,
                100,
                4.3,
                500,
                45,
                90,
                400
            )

    with t4:

        if st.button(
            "🛍️ Popular Product",
            use_container_width=True
        ):

            set_template(
                700,
                30,
                4.7,
                3000,
                55,
                150,
                800
            )


    st.divider()

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    st.subheader(
        "🎯 Product Analyzer"
    )

    left, right = st.columns(2)

    with left:

        price = st.number_input(
            "Product Price",
            min_value=0.0,
            max_value=100000.0,
            value=float(
                st.session_state.input_price
            ),
            step=10.0
        )

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            max_value=100000.0,
            value=float(
                st.session_state.input_discount
            ),
            step=5.0
        )

        rating = st.slider(
            "Product Rating",
            min_value=0.0,
            max_value=5.0,
            value=float(
                st.session_state.input_rating
            ),
            step=0.1
        )

        reviews = st.number_input(
            "Number of Reviews",
            min_value=0,
            max_value=1000000,
            value=int(
                st.session_state.input_reviews
            ),
            step=10
        )

    with right:

        title_length = st.number_input(
            "Title Length",
            min_value=0,
            max_value=1000,
            value=int(
                st.session_state.input_title_length
            ),
            step=1
        )

        feature_length = st.number_input(
            "Feature Length",
            min_value=0,
            max_value=5000,
            value=int(
                st.session_state.input_feature_length
            ),
            step=10
        )

        description_length = st.number_input(
            "Description Length",
            min_value=0,
            max_value=20000,
            value=int(
                st.session_state.input_description_length
            ),
            step=10
        )

    st.write("")

    analyze_col, reset_col = st.columns([3, 1])

    with analyze_col:

        analyze = st.button(
            "🚀 Analyze Product",
            use_container_width=True
        )

    with reset_col:

        if st.button(
            "↻ Reset",
            use_container_width=True
        ):

            set_template(
                500,
                50,
                4.5,
                1000,
                30,
                100,
                300
            )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if analyze:

        payload = {

            "price": price,
            "discount": discount,
            "rating": rating,
            "reviews": reviews,
            "title_length": title_length,
            "feature_length": feature_length,
            "description_length": description_length
        }

        progress = st.progress(
            0
        )

        status = st.empty()

        steps = [
            "⚙️ Preparing product features...",
            "📐 Applying StandardScaler...",
            "🧠 Running K-Means prediction...",
            "📊 Generating business insights..."
        ]

        for i, message in enumerate(steps):

            status.info(message)

            progress.progress(
                (i + 1) / len(steps)
            )

            time.sleep(0.2)

        response = predict_product(
            payload
        )

        progress.empty()
        status.empty()

        if response is None:

            st.error(
                "Could not communicate with FastAPI."
            )

        elif response.status_code == 200:

            result = response.json()

            prediction = result[
                "prediction"
            ]

            proximity = result[
                "cluster_proximity"
            ]

            insights = result.get(
                "insights",
                []
            )

            profile = result.get(
                "cluster_profile",
                {}
            )

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">

                <div style="
                    color:#94a3b8;
                    font-size:14px;
                ">
                PREDICTED PRODUCT SEGMENT
                </div>

                <div class="cluster-number">
                Cluster {prediction['cluster_id']}
                </div>

                <div class="cluster-name">
                {prediction['cluster_name']}
                </div>

                <p style="
                    color:#cbd5e1;
                    margin-top:12px;
                ">
                {prediction['description']}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Predicted Cluster",
                prediction["cluster_id"]
            )

            r2.metric(
                "Segment",
                prediction["cluster_name"]
            )

            r3.metric(
                "Proximity",
                f"{prediction.get('relative_proximity', 0):.2f}%"
            )

            # ------------------------------------------------
            # PROXIMITY
            # ------------------------------------------------

            st.subheader(
                "📏 Cluster Proximity"
            )

            st.caption(
                "Higher relative proximity means the product is closer to that cluster center."
            )

            for item in proximity:

                name = item["name"]

                pct = item[
                    "relative_proximity"
                ]

                distance = item[
                    "distance"
                ]

                st.write(
                    f"**{name}** — "
                    f"{pct:.2f}% proximity"
                )

                st.progress(
                    min(
                        max(
                            pct / 100,
                            0
                        ),
                        1
                    )
                )

                st.caption(
                    f"Distance: {distance:.4f}"
                )


            # ------------------------------------------------
            # BUSINESS INSIGHTS
            # ------------------------------------------------

            st.subheader(
                "💡 Business Insights"
            )

            for insight in insights:

                st.info(
                    f"💡 {insight}"
                )


            # ------------------------------------------------
            # PROFILE COMPARISON
            # ------------------------------------------------

            if profile:

                st.subheader(
                    "📊 Product vs Segment Average"
                )

                records = []

                labels = {

                    "price":
                        "Price",

                    "discount":
                        "Discount",

                    "rating":
                        "Rating",

                    "reviews":
                        "Reviews",

                    "title_length":
                        "Title Length",

                    "feature_length":
                        "Feature Length",

                    "description_length":
                        "Description Length"
                }

                for key, label in labels.items():

                    if key not in profile:
                        continue

                    user_value = profile[
                        key
                    ]["user_value"]

                    cluster_average = profile[
                        key
                    ]["cluster_average"]

                    records.append({

                        "Feature": label,

                        "Your Value":
                            round(
                                user_value,
                                2
                            ),

                        "Segment Average":
                            round(
                                cluster_average,
                                2
                            )
                    })

                st.dataframe(
                    pd.DataFrame(records),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.error(
                f"API Error: "
                f"{response.status_code}"
            )

            st.code(
                response.text
            )


# ============================================================
# SEGMENT ANALYTICS
# ============================================================

elif menu == "📊 Segment Analytics":

    st.title(
        "📊 Segment Analytics"
    )

    st.caption(
        "Real analytics generated from the processed grocery dataset."
    )

    if analytics_data is None:

        st.warning(
            "Analytics data could not be loaded."
        )

        st.stop()


    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    stats = analytics_data.get(
        "dataset_stats",
        overall_stats
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Total Products",
        f"{int(stats.get('total_products', 0)):,}"
    )

    a2.metric(
        "Average Price",
        f"₹{stats.get('avg_price', 0):,.2f}"
    )

    a3.metric(
        "Average Discount",
        f"{stats.get('avg_discount', 0):,.2f}"
    )

    a4.metric(
        "Average Rating",
        f"{stats.get('avg_rating', 0):.2f} ⭐"
    )

    st.divider()


    # --------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # --------------------------------------------------------

    distribution = analytics_data.get(
        "cluster_distribution",
        []
    )

    if distribution:

        dist_df = pd.DataFrame(
            distribution
        )

        left, right = st.columns(2)

        with left:

            fig = px.pie(
                dist_df,
                values="products",
                names="name",
                hole=0.55
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            fig = px.bar(
                dist_df,
                x="name",
                y="products",
                text="products"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # CLUSTER PROFILES
    # --------------------------------------------------------

    st.subheader(
        "📈 Cluster Profiles"
    )

    profiles = analytics_data.get(
        "cluster_profiles",
        {}
    )

    profile_rows = []

    for cluster_id, info in profiles.items():

        averages = info.get(
            "averages",
            {}
        )

        profile_rows.append({

            "Cluster":
                cluster_id,

            "Segment":
                info.get(
                    "name",
                    f"Cluster {cluster_id}"
                ),

            "Products":
                info.get(
                    "size",
                    0
                ),

            "Share %":
                info.get(
                    "percentage",
                    0
                ),

            "Avg Price":
                averages.get(
                    "price",
                    0
                ),

            "Avg Discount":
                averages.get(
                    "discount",
                    0
                ),

            "Avg Rating":
                averages.get(
                    "rating",
                    0
                ),

            "Avg Reviews":
                averages.get(
                    "reviews",
                    0
                )
        })

    if profile_rows:

        st.dataframe(
            pd.DataFrame(profile_rows),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EXPLORE CLUSTERS
# ============================================================

elif menu == "🧬 Explore Clusters":

    st.title(
        "🧬 Explore Product Segments"
    )

    st.caption(
        "Explore the characteristics of each discovered cluster."
    )

    profiles = analytics_data.get(
        "cluster_profiles",
        {}
    )

    if not profiles:

        st.warning(
            "Cluster profiles unavailable."
        )

        st.stop()


    tabs = st.tabs(
        [
            f"Cluster {cluster_id}"
            for cluster_id in profiles.keys()
        ]
    )


    for tab, (cluster_id, info) in zip(
        tabs,
        profiles.items()
    ):

        with tab:

            st.subheader(
                info.get(
                    "name",
                    f"Cluster {cluster_id}"
                )
            )

            st.write(
                info.get(
                    "description",
                    "Product segment discovered using K-Means clustering."
                )
            )

            averages = info.get(
                "averages",
                {}
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Products",
                f"{info.get('size', 0):,}"
            )

            c2.metric(
                "Dataset Share",
                f"{info.get('percentage', 0):.2f}%"
            )

            c3.metric(
                "Average Price",
                f"₹{averages.get('price', 0):,.2f}"
            )

            c4.metric(
                "Average Rating",
                f"{averages.get('rating', 0):.2f} ⭐"
            )

            st.write("")

            metric_df = pd.DataFrame({

                "Metric": [
                    "Price",
                    "Discount",
                    "Rating",
                    "Reviews",
                    "Title Length",
                    "Feature Length",
                    "Description Length"
                ],

                "Average": [

                    averages.get(
                        "price",
                        0
                    ),

                    averages.get(
                        "discount",
                        0
                    ),

                    averages.get(
                        "rating",
                        0
                    ),

                    averages.get(
                        "reviews",
                        0
                    ),

                    averages.get(
                        "title_length",
                        0
                    ),

                    averages.get(
                        "feature_length",
                        0
                    ),

                    averages.get(
                        "description_length",
                        0
                    )
                ]
            })

            fig = px.bar(
                metric_df,
                x="Metric",
                y="Average",
                text_auto=".2f"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif menu == "📖 About the Model":

    st.title(
        "📖 About the ML Model"
    )

    st.caption(
        "Understanding the machine learning pipeline behind AeroSegment."
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Algorithm",
        "K-Means"
    )

    m2.metric(
        "Features",
        len(features)
    )

    m3.metric(
        "Clusters",
        n_clusters
    )

    m4.metric(
        "Silhouette",
        evaluation.get(
            "silhouette_score",
            0
        )
    )

    st.divider()

    st.subheader(
        "🧠 Machine Learning Pipeline"
    )

    st.markdown(
        """
        **Dataset**

        ↓

        **EDA & Data Cleaning**

        ↓

        **Feature Engineering**

        ↓

        **Feature Selection**

        ↓

        **StandardScaler**

        ↓

        **K-Means Clustering**

        ↓

        **Cluster Evaluation**

        ↓

        **Product Segment Prediction**
        """
    )

    st.subheader(
        "📊 Evaluation Metrics"
    )

    e1, e2 = st.columns(2)

    with e1:

        st.metric(
            "Silhouette Score",
            evaluation.get(
                "silhouette_score",
                0
            )
        )

        st.caption(
            "Higher values generally indicate better cluster separation."
        )

    with e2:

        st.metric(
            "Davies-Bouldin Score",
            evaluation.get(
                "davies_bouldin_score",
                0
            )
        )

        st.caption(
            "Lower values generally indicate better cluster separation."
        )

    st.subheader(
        "🧮 Model Features"
    )

    for feature in features:

        st.write(
            f"• `{feature}`"
        )

    st.subheader(
        "🎯 Cluster Profiles"
    )

    for cluster_id, info in clusters_db.items():

        with st.expander(
            f"Cluster {cluster_id} — {info.get('name', '')}"
        ):

            st.write(
                info.get(
                    "description",
                    "K-Means product segment."
                )
            )

            st.write(
                "Products:",
                info.get(
                    "size",
                    0
                )
            )

            st.write(
                "Dataset Share:",
                f"{info.get('percentage', 0):.2f}%"
            )