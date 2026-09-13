from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "train.csv"
HERO_IMAGE_PATH = BASE_DIR / "Hero.png"

MODELS_DIR = BASE_DIR / "models"

LGB_MODEL_PATH = MODELS_DIR / "lightgbm.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
CAT_MODEL_PATH = MODELS_DIR / "catboost.cbm"

METADATA_PATH = MODELS_DIR / "model_metadata.json"


# ============================================================
# Project Configuration
# ============================================================

PIPELINE_VERSION = "1.1.0"

RANDOM_STATE = 42

TARGET_COLUMN = "Will_Buy_EV"


# Notebook 05 probability ensemble
BLEND_WEIGHTS = {
    "lightgbm": 0.55,
    "xgboost": 0.10,
    "catboost": 0.35,
}


RAW_INPUT_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


# Final Notebook-05 representation
SELECTED_RAW_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Income_x_Concern",
    "Subsidy_x_Income",
]


CATBOOST_CATEGORICAL_COLUMNS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
]


NUMERIC_INPUT_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]


INTEGER_INPUT_FEATURES = {
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
}


BINARY_MAP = {
    "No": 0,
    "Yes": 1,
}


ANXIETY_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}


# ============================================================
# Model Parameters
# ============================================================

LGB_PARAMS = {
    "objective": "binary",

    "learning_rate": 0.035,
    "n_estimators": 1800,

    "num_leaves": 32,
    "max_depth": 5,

    "min_child_samples": 62,

    "subsample": 0.8817235035948334,
    "colsample_bytree": 0.7422521041205801,

    "reg_alpha": 0.00010017150529609082,
    "reg_lambda": 2.9385486048550478,

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "verbosity": -1,
}


XGB_PARAMS = {
    "n_estimators": 1600,

    "learning_rate": 0.04,

    "max_depth": 7,

    "min_child_weight": 4,

    "subsample": 0.9,
    "colsample_bytree": 0.9,

    "reg_alpha": 0.15,
    "reg_lambda": 1.0,

    "objective": "binary:logistic",
    "eval_metric": "auc",

    "tree_method": "hist",

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "early_stopping_rounds": 80,
}


CAT_PARAMS = {
    "iterations": 1800,

    "learning_rate": 0.05,

    "depth": 5,

    "loss_function": "Logloss",
    "eval_metric": "AUC",

    "l2_leaf_reg": 3.6264901626422943,

    "random_strength": 1.951232084155201,

    "bagging_temperature": 0.35889771471518656,

    "random_seed": RANDOM_STATE,

    "verbose": False,

    "allow_writing_files": False,

    "thread_count": -1,
}


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


MODEL_STATE: dict[str, Any] = {
    "lightgbm": None,
    "xgboost": None,
    "catboost": None,
    "metadata": None,
}


# ============================================================
# Utility
# ============================================================

def log_step(message: str = "") -> None:
    print(
        message,
        flush=True,
    )


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def probability_tier(
    probability: float,
) -> str:

    if probability < 0.25:
        return "Low"

    if probability < 0.50:
        return "Moderate"

    if probability < 0.75:
        return "Elevated"

    return "High"


# ============================================================
# Training Data Validation
# ============================================================

def validate_training_schema(
    train_df: pd.DataFrame,
) -> None:

    required = set(
        RAW_INPUT_FEATURES
        +
        [TARGET_COLUMN]
    )

    missing = sorted(
        required
        -
        set(train_df.columns)
    )

    if missing:

        raise ValueError(
            "Training dataset is missing required columns: "
            +
            ", ".join(missing)
        )


def prepare_target(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_numeric_dtype(series):

        target = pd.to_numeric(
            series,
            errors="raise",
        ).astype(int)

    else:

        normalized = (
            series
            .astype(str)
            .str.strip()
            .str.lower()
        )

        target = normalized.map(
            {
                "no": 0,
                "yes": 1,
            }
        )

        if target.isna().any():

            unexpected = sorted(
                series[
                    target.isna()
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Unexpected target value(s): {unexpected}"
            )

        target = target.astype(int)


    unique_values = set(
        target.unique()
    )

    if not unique_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            "Target must contain only 0/1 or No/Yes."
        )

    return target


# ============================================================
# Metadata Helpers
# ============================================================

def create_form_metadata(
    train_df: pd.DataFrame,
) -> dict[str, Any]:

    numeric_ranges = {}

    for column in NUMERIC_INPUT_FEATURES:

        values = pd.to_numeric(
            train_df[column],
            errors="coerce",
        )

        numeric_ranges[column] = {
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
        }


    form_options = {}

    for column in [
        "Gender",
        "City_Type",
        "Current_Car_Type",
    ]:

        form_options[column] = sorted(
            train_df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


    form_options[
        "Home_Charging_Possible"
    ] = [
        "Yes",
        "No",
    ]

    form_options[
        "Subsidy_Available"
    ] = [
        "Yes",
        "No",
    ]

    form_options[
        "Range_Anxiety_Level"
    ] = [
        "Low",
        "Medium",
        "High",
    ]


    return {
        "numeric_ranges": numeric_ranges,
        "form_options": form_options,
    }


def load_metadata() -> dict[str, Any]:

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def models_are_current(
    dataset_sha256: str,
) -> bool:

    required_files = [
        LGB_MODEL_PATH,
        XGB_MODEL_PATH,
        CAT_MODEL_PATH,
        METADATA_PATH,
    ]

    if not all(
        path.exists()
        for path in required_files
    ):

        return False


    try:

        metadata = load_metadata()

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return False


    return (
        metadata.get(
            "pipeline_version"
        )
        ==
        PIPELINE_VERSION

        and

        metadata.get(
            "dataset_sha256"
        )
        ==
        dataset_sha256

        and

        isinstance(
            metadata.get(
                "tree_feature_columns"
            ),
            list,
        )

        and

        len(
            metadata.get(
                "tree_feature_columns",
                [],
            )
        )
        >
        0
    )


# ============================================================
# Feature Engineering
# ============================================================

def build_features(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:

    missing = [
        column
        for column in RAW_INPUT_FEATURES
        if column not in raw_df.columns
    ]

    if missing:

        raise ValueError(
            "Input data is missing columns: "
            +
            ", ".join(missing)
        )


    out = raw_df[
        RAW_INPUT_FEATURES
    ].copy()


    # Binary values
    out[
        "Home_Charging_Possible"
    ] = out[
        "Home_Charging_Possible"
    ].map(
        BINARY_MAP
    )


    out[
        "Subsidy_Available"
    ] = out[
        "Subsidy_Available"
    ].map(
        BINARY_MAP
    )


    # Ordinal range anxiety
    out[
        "Range_Anxiety_Level"
    ] = out[
        "Range_Anxiety_Level"
    ].map(
        ANXIETY_MAP
    )


    mapped_columns = [
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]


    if out[
        mapped_columns
    ].isna().any().any():

        raise ValueError(
            "Unexpected categorical value in feature pipeline."
        )


    out[
        mapped_columns
    ] = out[
        mapped_columns
    ].astype(int)


    # Notebook-05 accepted interaction
    out[
        "Income_x_Concern"
    ] = (
        (
            out[
                "Annual_Income_USD"
            ].astype(float)
            /
            10000.0
        )
        *
        out[
            "Environmental_Concern_Level"
        ].astype(float)
    )


    # Notebook-05 accepted interaction
    out[
        "Subsidy_x_Income"
    ] = (
        out[
            "Subsidy_Available"
        ].astype(float)
        *
        (
            out[
                "Annual_Income_USD"
            ].astype(float)
            /
            10000.0
        )
    )


    return out[
        SELECTED_RAW_FEATURES
    ].copy()


# ============================================================
# Tree Encoding
# ============================================================

def encode_tree_features(
    selected_df: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:

    frame = selected_df.copy()


    categorical_columns = [
        column
        for column
        in CATBOOST_CATEGORICAL_COLUMNS
        if column in frame.columns
    ]


    frame = pd.get_dummies(
        frame,
        columns=categorical_columns,
        drop_first=False,
        dtype=np.int8,
    )


    if expected_columns is not None:

        frame = frame.reindex(
            columns=expected_columns,
            fill_value=0,
        )


    return frame


def prepare_catboost_features(
    selected_df: pd.DataFrame,
) -> pd.DataFrame:

    frame = selected_df.copy()

    for column in CATBOOST_CATEGORICAL_COLUMNS:

        frame[column] = (
            frame[column]
            .astype(str)
        )

    return frame


# ============================================================
# Prediction Validation
# ============================================================

def validate_payload(
    payload: Any,
) -> dict[str, Any]:

    if not isinstance(
        payload,
        dict,
    ):

        raise ValueError(
            "Request body must be a JSON object."
        )


    missing = [
        feature
        for feature in RAW_INPUT_FEATURES
        if feature not in payload
    ]

    if missing:

        raise ValueError(
            "Missing required field(s): "
            +
            ", ".join(missing)
        )


    metadata = (
        MODEL_STATE.get(
            "metadata"
        )
        or
        {}
    )


    ranges = metadata.get(
        "numeric_ranges",
        {},
    )


    options = metadata.get(
        "form_options",
        {},
    )


    clean = {}


    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for field in NUMERIC_INPUT_FEATURES:

        try:

            value = float(
                payload[field]
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"{field} must be numeric."
            ) from None


        if not np.isfinite(value):

            raise ValueError(
                f"{field} must be finite."
            )


        field_range = ranges.get(
            field
        )


        if field_range:

            minimum = float(
                field_range["min"]
            )

            maximum = float(
                field_range["max"]
            )


            if (
                value < minimum
                or
                value > maximum
            ):

                raise ValueError(
                    f"{field} must be between "
                    f"{minimum:g} and {maximum:g}."
                )


        if field in INTEGER_INPUT_FEATURES:

            if not value.is_integer():

                raise ValueError(
                    f"{field} must be a whole number."
                )

            clean[field] = int(
                value
            )

        else:

            clean[field] = float(
                value
            )


    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    categorical_fields = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]


    for field in categorical_fields:

        value = str(
            payload[field]
        ).strip()


        allowed = options.get(
            field,
            [],
        )


        if (
            allowed
            and
            value not in allowed
        ):

            raise ValueError(
                f"Invalid {field} value."
            )


        clean[field] = value


    return clean


# ============================================================
# XGBoost Iteration Helper
# ============================================================

def get_xgb_best_iteration(
    model: XGBClassifier,
    fallback: int,
) -> int:

    try:

        best_iteration = int(
            model.best_iteration
        )

        return max(
            best_iteration + 1,
            50,
        )

    except Exception:

        return fallback


# ============================================================
# First-Time Training
# ============================================================

def train_and_save_models(
    dataset_sha256: str,
) -> None:

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "\nTraining dataset not found.\n"
            f"Expected: {DATA_PATH}\n"
            "Place train.csv inside data/train.csv"
        )


    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Metadata file acts as bundle completion marker.
    METADATA_PATH.unlink(
        missing_ok=True
    )


    log_step()
    log_step(
        "VoltVision Initialization"
    )
    log_step(
        "─" * 50
    )
    log_step(
        "Loading training dataset..."
    )


    train_df = pd.read_csv(
        DATA_PATH
    )


    validate_training_schema(
        train_df
    )


    y = prepare_target(
        train_df[
            TARGET_COLUMN
        ]
    )


    form_metadata = create_form_metadata(
        train_df
    )


    selected = build_features(
        train_df
    )


    tree_X = encode_tree_features(
        selected
    )


    cat_X = prepare_catboost_features(
        selected
    )


    cat_feature_indices = [
        cat_X.columns.get_loc(
            column
        )
        for column
        in CATBOOST_CATEGORICAL_COLUMNS
    ]


    log_step(
        f"✓ Dataset loaded: {len(train_df):,} rows"
    )

    log_step(
        f"✓ Final tree features: {tree_X.shape[1]}"
    )

    log_step(
        "✓ Shared feature pipeline prepared"
    )

    log_step()
    log_step(
        "Production models do not exist yet."
    )

    log_step(
        "Starting one-time model training..."
    )


    # ========================================================
    # Validation Split
    # ========================================================

    indices = np.arange(
        len(y)
    )


    train_idx, valid_idx = train_test_split(
        indices,

        test_size=0.10,

        random_state=RANDOM_STATE,

        stratify=y,
    )


    validation_scores = {}


    # ========================================================
    # LightGBM Probe
    # ========================================================

    log_step()
    log_step(
        "[1/3] LightGBM early-stopping probe..."
    )


    lgb_probe = LGBMClassifier(
        **LGB_PARAMS
    )


    lgb_probe.fit(
        tree_X.iloc[
            train_idx
        ],

        y.iloc[
            train_idx
        ],

        eval_set=[
            (
                tree_X.iloc[
                    valid_idx
                ],

                y.iloc[
                    valid_idx
                ],
            )
        ],

        eval_metric="auc",

        callbacks=[
            early_stopping(
                100,
                verbose=False,
            ),

            log_evaluation(0),
        ],
    )


    lgb_validation_probability = (
        lgb_probe.predict_proba(
            tree_X.iloc[
                valid_idx
            ]
        )[:, 1]
    )


    validation_scores[
        "lightgbm"
    ] = float(
        roc_auc_score(
            y.iloc[
                valid_idx
            ],

            lgb_validation_probability,
        )
    )


    lgb_iterations = int(
        getattr(
            lgb_probe,
            "best_iteration_",
            0,
        )
        or
        LGB_PARAMS[
            "n_estimators"
        ]
    )


    lgb_iterations = max(
        lgb_iterations,
        50,
    )


    log_step(
        f"✓ LightGBM AUC: "
        f"{validation_scores['lightgbm']:.6f}"
    )

    log_step(
        f"✓ Best iteration: {lgb_iterations}"
    )


    # ========================================================
    # XGBoost Probe
    # ========================================================

    log_step()
    log_step(
        "[2/3] XGBoost early-stopping probe..."
    )


    xgb_probe = XGBClassifier(
        **XGB_PARAMS
    )


    xgb_probe.fit(
        tree_X.iloc[
            train_idx
        ],

        y.iloc[
            train_idx
        ],

        eval_set=[
            (
                tree_X.iloc[
                    valid_idx
                ],

                y.iloc[
                    valid_idx
                ],
            )
        ],

        verbose=False,
    )


    xgb_validation_probability = (
        xgb_probe.predict_proba(
            tree_X.iloc[
                valid_idx
            ]
        )[:, 1]
    )


    validation_scores[
        "xgboost"
    ] = float(
        roc_auc_score(
            y.iloc[
                valid_idx
            ],

            xgb_validation_probability,
        )
    )


    xgb_iterations = get_xgb_best_iteration(
        xgb_probe,
        XGB_PARAMS[
            "n_estimators"
        ],
    )


    log_step(
        f"✓ XGBoost AUC: "
        f"{validation_scores['xgboost']:.6f}"
    )

    log_step(
        f"✓ Best iteration: {xgb_iterations}"
    )


    # ========================================================
    # CatBoost Probe
    # ========================================================

    log_step()
    log_step(
        "[3/3] CatBoost early-stopping probe..."
    )


    cat_probe = CatBoostClassifier(
        **CAT_PARAMS
    )


    cat_probe.fit(
        cat_X.iloc[
            train_idx
        ],

        y.iloc[
            train_idx
        ],

        cat_features=cat_feature_indices,

        eval_set=(
            cat_X.iloc[
                valid_idx
            ],

            y.iloc[
                valid_idx
            ],
        ),

        early_stopping_rounds=100,

        verbose=False,
    )


    cat_validation_probability = (
        cat_probe.predict_proba(
            cat_X.iloc[
                valid_idx
            ]
        )[:, 1]
    )


    validation_scores[
        "catboost"
    ] = float(
        roc_auc_score(
            y.iloc[
                valid_idx
            ],

            cat_validation_probability,
        )
    )


    cat_best_iteration = int(
        cat_probe.get_best_iteration()
    )


    if cat_best_iteration >= 0:

        cat_iterations = (
            cat_best_iteration
            +
            1
        )

    else:

        cat_iterations = CAT_PARAMS[
            "iterations"
        ]


    cat_iterations = max(
        cat_iterations,
        50,
    )


    log_step(
        f"✓ CatBoost AUC: "
        f"{validation_scores['catboost']:.6f}"
    )

    log_step(
        f"✓ Best iteration: {cat_iterations}"
    )


    # ========================================================
    # Retrain on 100%
    # ========================================================

    log_step()
    log_step(
        "Retraining production ensemble on 100% data..."
    )


    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    final_lgb_params = dict(
        LGB_PARAMS
    )

    final_lgb_params[
        "n_estimators"
    ] = lgb_iterations


    lightgbm_model = LGBMClassifier(
        **final_lgb_params
    )


    lightgbm_model.fit(
        tree_X,
        y,
    )


    log_step(
        "✓ Final LightGBM trained"
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    final_xgb_params = dict(
        XGB_PARAMS
    )

    final_xgb_params[
        "n_estimators"
    ] = xgb_iterations


    final_xgb_params.pop(
        "early_stopping_rounds",
        None,
    )


    xgboost_model = XGBClassifier(
        **final_xgb_params
    )


    xgboost_model.fit(
        tree_X,
        y,
        verbose=False,
    )


    log_step(
        "✓ Final XGBoost trained"
    )


    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    final_cat_params = dict(
        CAT_PARAMS
    )

    final_cat_params[
        "iterations"
    ] = cat_iterations


    catboost_model = CatBoostClassifier(
        **final_cat_params
    )


    catboost_model.fit(
        cat_X,
        y,

        cat_features=cat_feature_indices,

        verbose=False,
    )


    log_step(
        "✓ Final CatBoost trained"
    )


    # ========================================================
    # Save Models
    # ========================================================

    log_step()
    log_step(
        "Saving production model bundle..."
    )


    joblib.dump(
        lightgbm_model,
        LGB_MODEL_PATH,
    )


    joblib.dump(
        xgboost_model,
        XGB_MODEL_PATH,
    )


    catboost_model.save_model(
        str(
            CAT_MODEL_PATH
        )
    )


    metadata = {
        "pipeline_version":
            PIPELINE_VERSION,

        "trained_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "dataset_sha256":
            dataset_sha256,

        "training_rows":
            int(
                len(train_df)
            ),

        "target":
            TARGET_COLUMN,

        "raw_input_features":
            RAW_INPUT_FEATURES,

        "selected_raw_features":
            SELECTED_RAW_FEATURES,

        "tree_feature_columns":
            tree_X.columns.tolist(),

        "catboost_columns":
            cat_X.columns.tolist(),

        "catboost_categorical_columns":
            CATBOOST_CATEGORICAL_COLUMNS,

        "blend_weights":
            BLEND_WEIGHTS,

        "numeric_ranges":
            form_metadata[
                "numeric_ranges"
            ],

        "form_options":
            form_metadata[
                "form_options"
            ],

        "iterations": {
            "lightgbm":
                int(
                    lgb_iterations
                ),

            "xgboost":
                int(
                    xgb_iterations
                ),

            "catboost":
                int(
                    cat_iterations
                ),
        },

        "probe_validation_auc":
            validation_scores,

        "notebook_reference": {
            "oof_probability_blend_auc":
                0.942399,

            "oof_probability_blend_weights": [
                0.55,
                0.10,
                0.35,
            ],
        },
    }


    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    log_step(
        "✓ Models saved"
    )

    log_step(
        "✓ Metadata saved"
    )


# ============================================================
# Model Loading
# ============================================================

def load_models() -> None:

    MODEL_STATE[
        "lightgbm"
    ] = joblib.load(
        LGB_MODEL_PATH
    )


    MODEL_STATE[
        "xgboost"
    ] = joblib.load(
        XGB_MODEL_PATH
    )


    catboost_model = CatBoostClassifier()


    catboost_model.load_model(
        str(
            CAT_MODEL_PATH
        )
    )


    MODEL_STATE[
        "catboost"
    ] = catboost_model


    MODEL_STATE[
        "metadata"
    ] = load_metadata()


def model_state_ready() -> bool:

    return all(
        MODEL_STATE.get(
            key
        )
        is not None

        for key in [
            "lightgbm",
            "xgboost",
            "catboost",
            "metadata",
        ]
    )


# ============================================================
# Initialization
# ============================================================

def initialize_models() -> None:

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "\nTraining dataset not found.\n"
            f"Expected:\n{DATA_PATH}"
        )


    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    dataset_sha256 = sha256_file(
        DATA_PATH
    )


    if models_are_current(
        dataset_sha256
    ):

        log_step()
        log_step(
            "VoltVision Initialization"
        )
        log_step(
            "─" * 50
        )

        log_step(
            "✓ Existing compatible model bundle detected"
        )


        try:

            load_models()

            log_step(
                "✓ Models loaded"
            )

            return

        except Exception as error:

            log_step(
                f"! Model loading failed: {error}"
            )

            log_step(
                "Rebuilding model bundle..."
            )


    train_and_save_models(
        dataset_sha256
    )


    load_models()


# ============================================================
# Ensemble Prediction
# ============================================================

def predict_ensemble(
    payload: dict[str, Any],
) -> dict[str, Any]:

    if not model_state_ready():

        raise RuntimeError(
            "Model bundle is not initialized."
        )


    clean_payload = validate_payload(
        payload
    )


    raw_row = pd.DataFrame(
        [clean_payload],

        columns=RAW_INPUT_FEATURES,
    )


    selected = build_features(
        raw_row
    )


    metadata = MODEL_STATE[
        "metadata"
    ]


    # --------------------------------------------------------
    # Tree representation
    # --------------------------------------------------------

    tree_row = encode_tree_features(
        selected,

        expected_columns=metadata[
            "tree_feature_columns"
        ],
    )


    # --------------------------------------------------------
    # Native CatBoost representation
    # --------------------------------------------------------

    cat_row = prepare_catboost_features(
        selected
    )


    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    lgb_probability = float(
        MODEL_STATE[
            "lightgbm"
        ]
        .predict_proba(
            tree_row
        )[0, 1]
    )


    xgb_probability = float(
        MODEL_STATE[
            "xgboost"
        ]
        .predict_proba(
            tree_row
        )[0, 1]
    )


    cat_probability = float(
        MODEL_STATE[
            "catboost"
        ]
        .predict_proba(
            cat_row
        )[0, 1]
    )


    weights = metadata.get(
        "blend_weights",
        BLEND_WEIGHTS,
    )


    ensemble_probability = (
        float(
            weights[
                "lightgbm"
            ]
        )
        *
        lgb_probability

        +

        float(
            weights[
                "xgboost"
            ]
        )
        *
        xgb_probability

        +

        float(
            weights[
                "catboost"
            ]
        )
        *
        cat_probability
    )


    ensemble_probability = float(
        np.clip(
            ensemble_probability,
            0.0,
            1.0,
        )
    )


    return {
        "probability":
            round(
                ensemble_probability,
                6,
            ),

        "percentage":
            round(
                ensemble_probability
                *
                100,
                2,
            ),

        "tier":
            probability_tier(
                ensemble_probability
            ),

        "models": {
            "lightgbm":
                round(
                    lgb_probability,
                    6,
                ),

            "xgboost":
                round(
                    xgb_probability,
                    6,
                ),

            "catboost":
                round(
                    cat_probability,
                    6,
                ),
        },

        "weights":
            weights,
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def index():

    metadata = (
        MODEL_STATE.get(
            "metadata"
        )
        or
        {}
    )


    return render_template(
        "index.html",

        form_options=metadata.get(
            "form_options",
            {},
        ),

        numeric_ranges=metadata.get(
            "numeric_ranges",
            {},
        ),
    )


# Hero image is intentionally stored in project root.
@app.get("/Hero.png")
def hero_image():

    if not HERO_IMAGE_PATH.exists():

        return (
            "Hero image not found.",
            404,
        )


    return send_from_directory(
        str(
            BASE_DIR
        ),

        HERO_IMAGE_PATH.name,
    )


@app.get("/health")
def health():

    metadata = (
        MODEL_STATE.get(
            "metadata"
        )
        or
        {}
    )


    return jsonify(
        {
            "status":
                (
                    "ok"
                    if model_state_ready()
                    else "initializing"
                ),

            "models_ready":
                model_state_ready(),

            "pipeline_version":
                metadata.get(
                    "pipeline_version",
                    PIPELINE_VERSION,
                ),
        }
    )


@app.post("/predict")
def predict():

    try:

        payload = request.get_json(
            silent=True
        )


        result = predict_ensemble(
            payload
        )


        return jsonify(
            {
                "ok": True,
                **result,
            }
        )


    except ValueError as error:

        return jsonify(
            {
                "ok": False,
                "error":
                    str(error),
            }
        ), 400


    except RuntimeError as error:

        return jsonify(
            {
                "ok": False,
                "error":
                    str(error),
            }
        ), 503


    except Exception:

        app.logger.exception(
            "Prediction failed"
        )


        return jsonify(
            {
                "ok": False,
                "error":
                    "Prediction failed because of an internal server error.",
            }
        ), 500


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        initialize_models()

    except Exception as error:

        print(
            "\nERROR: VoltVision could not initialize.",
            file=sys.stderr,
        )

        print(
            str(error),
            file=sys.stderr,
        )

        sys.exit(1)


    if not HERO_IMAGE_PATH.exists():

        log_step()
        log_step(
            "! Warning: Hero.png was not found in project root."
        )


    host = os.getenv(
        "VOLTVISION_HOST",
        "127.0.0.1",
    )


    port = int(
        os.getenv(
            "VOLTVISION_PORT",
            "5000",
        )
    )


    log_step()
    log_step(
        "✓ VoltVision ready"
    )

    log_step(
        f"✓ http://{host}:{port}"
    )

    log_step(
        "─" * 50
    )


    app.run(
        host=host,

        port=port,

        debug=False,

        use_reloader=False,
    )