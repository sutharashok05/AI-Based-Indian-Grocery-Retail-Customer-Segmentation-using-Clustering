from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import ProductInput


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "kmeans.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"
DATA_PATH = BASE_DIR / "data" / "GroceryDataset_EDA_Final.csv"


# ============================================================
# LOAD EXISTING ML ARTIFACTS
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

except Exception as e:
    raise RuntimeError(
        f"Failed to load ML artifacts: {e}"
    )


# ============================================================
# VALIDATE ML ARTIFACTS
# ============================================================

if not hasattr(model, "predict"):
    raise RuntimeError(
        "Invalid K-Means model."
    )

if not hasattr(model, "transform"):
    raise RuntimeError(
        "Loaded model does not support transform()."
    )

if not hasattr(scaler, "transform"):
    raise RuntimeError(
        "Invalid scaler."
    )

if not isinstance(features, list):
    features = list(features)


# ============================================================
# LOAD FINAL DATASET
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)

except Exception as e:
    raise RuntimeError(
        f"Failed to load dataset: {e}"
    )


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    raise RuntimeError(
        f"Missing model features in dataset: "
        f"{missing_features}"
    )


# ============================================================
# PREPARE DATA FOR CLUSTER ANALYTICS
# ============================================================

X = df[features].copy()

# Convert values to numeric
for column in features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# Replace infinite values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill missing values using median
for column in features:

    median_value = X[column].median()

    if pd.isna(median_value):
        median_value = 0.0

    X[column] = X[column].fillna(
        median_value
    )


# ============================================================
# SCALE USING EXISTING SCALER
# ============================================================

try:
    X_scaled = scaler.transform(X)

except Exception as e:
    raise RuntimeError(
        f"Scaler transformation failed: {e}"
    )


# ============================================================
# ASSIGN EXISTING K-MEANS CLUSTERS
# ============================================================

try:
    df["Cluster"] = model.predict(
        X_scaled
    ).astype(int)

except Exception as e:
    raise RuntimeError(
        f"K-Means prediction failed: {e}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_mean(column):
    """
    Safely calculate mean of a dataset column.
    """

    if column not in df.columns:
        return 0.0

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    value = values.mean()

    if pd.isna(value):
        return 0.0

    return round(
        float(value),
        2
    )


def cluster_mean(
    cluster_df,
    column
):
    """
    Safely calculate mean for one cluster.
    """

    if column not in cluster_df.columns:
        return 0.0

    values = pd.to_numeric(
        cluster_df[column],
        errors="coerce"
    )

    value = values.mean()

    if pd.isna(value):
        return 0.0

    return round(
        float(value),
        2
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

DATASET_STATS = {

    "total_products":
        int(len(df)),

    "avg_price":
        safe_mean("Price_num"),

    "avg_rating":
        safe_mean("Rating_num"),

    "avg_discount":
        safe_mean("Discount_num"),

    "avg_reviews":
        safe_mean("Reviews_num")
}


# ============================================================
# CLUSTER IDS
# ============================================================

cluster_ids = sorted(
    df["Cluster"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)


# ============================================================
# BUILD CLUSTER INFORMATION
# ============================================================

CLUSTER_INFO = {}


for cluster_id in cluster_ids:

    cluster_df = df[
        df["Cluster"] == cluster_id
    ].copy()

    cluster_size = len(
        cluster_df
    )

    cluster_percentage = (
        cluster_size / len(df) * 100
        if len(df) > 0
        else 0
    )

    avg_price = cluster_mean(
        cluster_df,
        "Price_num"
    )

    avg_discount = cluster_mean(
        cluster_df,
        "Discount_num"
    )

    avg_rating = cluster_mean(
        cluster_df,
        "Rating_num"
    )

    avg_reviews = cluster_mean(
        cluster_df,
        "Reviews_num"
    )

    # --------------------------------------------------------
    # Dynamic segment description
    # --------------------------------------------------------

    if avg_price >= DATASET_STATS["avg_price"]:

        price_description = (
            "higher-priced products"
        )

    else:

        price_description = (
            "lower-priced products"
        )


    if avg_reviews >= DATASET_STATS["avg_reviews"]:

        engagement_description = (
            "strong customer engagement"
        )

    else:

        engagement_description = (
            "relatively lower customer engagement"
        )


    if avg_discount >= DATASET_STATS["avg_discount"]:

        discount_description = (
            "higher discount levels"
        )

    else:

        discount_description = (
            "lower discount levels"
        )


    description = (
        f"This segment contains {cluster_size:,} "
        f"products ({cluster_percentage:.2f}% of the dataset). "
        f"It generally represents {price_description}, "
        f"{discount_description}, and "
        f"{engagement_description}."
    )


    # --------------------------------------------------------
    # Cluster name
    # --------------------------------------------------------

    if (
        avg_price < DATASET_STATS["avg_price"]
        and
        avg_reviews < DATASET_STATS["avg_reviews"]
    ):

        cluster_name = (
            f"Value Segment {cluster_id}"
        )

    elif (
        avg_price >= DATASET_STATS["avg_price"]
        and
        avg_reviews >= DATASET_STATS["avg_reviews"]
    ):

        cluster_name = (
            f"Premium Popular Segment {cluster_id}"
        )

    elif avg_price >= DATASET_STATS["avg_price"]:

        cluster_name = (
            f"Premium Segment {cluster_id}"
        )

    else:

        cluster_name = (
            f"Value Segment {cluster_id}"
        )


    CLUSTER_INFO[str(cluster_id)] = {

        "name":
            cluster_name,

        "description":
            description,

        "size":
            int(cluster_size),

        "percentage":
            round(
                float(cluster_percentage),
                2
            ),

        "averages": {

            "price":
                avg_price,

            "discount":
                avg_discount,

            "rating":
                avg_rating,

            "reviews":
                avg_reviews,

            "title_length":
                cluster_mean(
                    cluster_df,
                    "Title_Length"
                ),

            "feature_length":
                cluster_mean(
                    cluster_df,
                    "Feature_Length"
                ),

            "description_length":
                cluster_mean(
                    cluster_df,
                    "Description_Length"
                )
        }
    }


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "AI-Based Indian Grocery "
        "Segmentation API"
    ),

    description=(
        "FastAPI backend for grocery "
        "product segmentation using "
        "an existing K-Means clustering model."
    ),

    version="2.1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=[
        "*"
    ]
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():

    return {

        "success": True,

        "message":
            "AI Grocery Segmentation API is running",

        "algorithm":
            "K-Means Clustering",

        "status":
            "ready",

        "version":
            "2.1.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model_loaded":
            model is not None,

        "scaler_loaded":
            scaler is not None,

        "features_loaded":
            features is not None,

        "number_of_features":
            len(features),

        "number_of_clusters":
            int(model.n_clusters),

        "dataset_rows":
            int(len(df))
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "algorithm":
            "K-Means Clustering",

        "number_of_clusters":
            int(model.n_clusters),

        "features":
            features,

        "dataset": {

            "total_products":
                int(len(df)),

            "statistics":
                DATASET_STATS
        },

        "evaluation": {

            "silhouette_score":
                0.4314,

            "davies_bouldin_score":
                1.6301
        },

        "clusters":
            CLUSTER_INFO
    }


# ============================================================
# ANALYTICS DATA
# ============================================================

@app.get("/analytics-data")
def analytics_data():

    cluster_distribution = []

    for cluster_id in cluster_ids:

        cluster_df = df[
            df["Cluster"] == cluster_id
        ]

        size = len(
            cluster_df
        )

        percentage = (
            size / len(df) * 100
            if len(df) > 0
            else 0
        )

        cluster_distribution.append({

            "cluster_id":
                int(cluster_id),

            "name":
                CLUSTER_INFO[
                    str(cluster_id)
                ]["name"],

            "products":
                int(size),

            "percentage":
                round(
                    float(percentage),
                    2
                )
        })


    return {

        "success":
            True,

        "count":
            int(len(df)),

        "dataset_stats":
            DATASET_STATS,

        "cluster_distribution":
            cluster_distribution,

        "cluster_profiles":
            CLUSTER_INFO
    }


# ============================================================
# ANALYTICS ALIAS
# ============================================================

@app.get("/analytics")
def analytics():

    return analytics_data()


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict_cluster(
    product: ProductInput
):

    try:

        # ----------------------------------------------------
        # Create input dictionary
        # ----------------------------------------------------

        input_values = {

            "Price_num":
                product.price,

            "Discount_num":
                product.discount,

            "Rating_num":
                product.rating,

            "Reviews_num":
                product.reviews,

            "Title_Length":
                product.title_length,

            "Feature_Length":
                product.feature_length,

            "Description_Length":
                product.description_length
        }


        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [input_values]
        )


        # ----------------------------------------------------
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

        input_data = input_data[
            features
        ]


        # ----------------------------------------------------
        # Safety conversion
        # ----------------------------------------------------

        input_data = input_data.astype(
            float
        )


        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        scaled_data = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # K-MEANS PREDICTION
        # ----------------------------------------------------

        cluster_id = int(
            model.predict(
                scaled_data
            )[0]
        )


        # ----------------------------------------------------
        # DISTANCE TO ALL CLUSTERS
        # ----------------------------------------------------

        distances = model.transform(
            scaled_data
        )[0]


        # ----------------------------------------------------
        # RELATIVE PROXIMITY
        #
        # IMPORTANT:
        # This is NOT probability/confidence.
        # It represents relative closeness.
        # ----------------------------------------------------

        epsilon = 1e-8

        inverse_distance = (
            1.0 /
            (
                distances +
                epsilon
            )
        )

        total_inverse_distance = (
            inverse_distance.sum()
        )

        if total_inverse_distance > 0:

            proximity = (
                inverse_distance /
                total_inverse_distance *
                100
            )

        else:

            proximity = np.zeros(
                len(distances)
            )


        # ----------------------------------------------------
        # CLUSTER INFORMATION
        # ----------------------------------------------------

        cluster_info = CLUSTER_INFO.get(
            str(cluster_id),
            {}
        )


        # ----------------------------------------------------
        # CLUSTER PROXIMITY RESPONSE
        # ----------------------------------------------------

        cluster_proximity = []

        for index, distance in enumerate(
            distances
        ):

            cluster_data = CLUSTER_INFO.get(
                str(index),
                {}
            )

            cluster_proximity.append({

                "cluster_id":
                    int(index),

                "name":
                    cluster_data.get(
                        "name",
                        f"Cluster {index}"
                    ),

                "distance":
                    round(
                        float(distance),
                        4
                    ),

                "relative_proximity":
                    round(
                        float(
                            proximity[index]
                        ),
                        2
                    )
            })


        # ----------------------------------------------------
        # CLUSTER AVERAGES
        # ----------------------------------------------------

        cluster_average = (
            cluster_info
            .get(
                "averages",
                {}
            )
        )


        # ----------------------------------------------------
        # PROFILE COMPARISON
        # ----------------------------------------------------

        feature_mapping = {

            "price":
                "price",

            "discount":
                "discount",

            "rating":
                "rating",

            "reviews":
                "reviews",

            "title_length":
                "title_length",

            "feature_length":
                "feature_length",

            "description_length":
                "description_length"
        }


        profile = {}


        for input_name, average_name in (
            feature_mapping.items()
        ):

            user_value = getattr(
                product,
                input_name
            )

            average_value = cluster_average.get(
                average_name,
                0
            )


            # ----------------------------------------------
            # Percentage deviation
            # ----------------------------------------------

            if average_value != 0:

                deviation_percent = (
                    (
                        float(user_value)
                        - float(average_value)
                    )
                    /
                    abs(float(average_value))
                    *
                    100
                )

            else:

                deviation_percent = 0.0


            profile[input_name] = {

                "user_value":
                    round(
                        float(user_value),
                        2
                    ),

                "cluster_average":
                    round(
                        float(average_value),
                        2
                    ),

                "deviation_percent":
                    round(
                        float(
                            deviation_percent
                        ),
                        2
                    )
            }


        # ----------------------------------------------------
        # BUSINESS INSIGHTS
        # ----------------------------------------------------

        insights = []


        # PRICE
        price_average = cluster_average.get(
            "price",
            0
        )

        if price_average > 0:

            price_difference = (
                product.price
                -
                price_average
            ) / price_average * 100


            if price_difference > 15:

                insights.append(
                    "The product price is "
                    "significantly above the "
                    "assigned segment average."
                )

            elif price_difference < -15:

                insights.append(
                    "The product price is "
                    "significantly below the "
                    "assigned segment average."
                )

            else:

                insights.append(
                    "The product price is aligned "
                    "with the assigned segment."
                )


        # RATING
        rating_average = cluster_average.get(
            "rating",
            0
        )

        if product.rating >= rating_average:

            insights.append(
                "The product rating is at or "
                "above the segment average."
            )

        else:

            insights.append(
                "The product rating is below "
                "the segment average."
            )


        # REVIEWS
        reviews_average = cluster_average.get(
            "reviews",
            0
        )

        if product.reviews >= reviews_average:

            insights.append(
                "The product has relatively "
                "strong customer engagement."
            )

        else:

            insights.append(
                "The product has lower review "
                "volume than the segment average."
            )


        # DISCOUNT
        discount_average = cluster_average.get(
            "discount",
            0
        )

        if product.discount > discount_average:

            insights.append(
                "The product offers a higher "
                "discount than the segment average."
            )

        else:

            insights.append(
                "The product discount is at or "
                "below the segment average."
            )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        selected_proximity = 0.0

        if (
            0 <= cluster_id
            < len(proximity)
        ):

            selected_proximity = float(
                proximity[cluster_id]
            )


        return {

            "success":
                True,

            "prediction": {

                "cluster_id":
                    cluster_id,

                "cluster_name":
                    cluster_info.get(
                        "name",
                        f"Product Segment {cluster_id}"
                    ),

                "description":
                    cluster_info.get(
                        "description",
                        "Product segment identified "
                        "using K-Means clustering."
                    ),

                "relative_proximity":
                    round(
                        selected_proximity,
                        2
                    )
            },

            "cluster_proximity":
                cluster_proximity,

            "cluster_profile":
                profile,

            "insights":
                insights
        }


    except Exception as e:

        raise HTTPException(

            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail={

                "success":
                    False,

                "message":
                    "Prediction failed.",

                "error":
                    str(e)
            }
        )