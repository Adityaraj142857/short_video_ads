"""
data_loader.py
==============
Handles raw data ingestion, cleaning, encoding, and
derived feature engineering for the short-form video /
purchase-intent analysis project.

Column index reference (0-based after Timestamp):
  0  Timestamp
  1  Age
  2  Gender
  3  Occupation
  4  Social media hours/day
  5  Primary platform
  6  Purchase considerations (multi-select text)
  7  Liked product
  8  Best features product
  9  Best value product
 10  Influencer knowledgeable          (Likert 1-5)
 11  Influencer trustworthy            (Likert 1-5)
 12  General trust in tech influencers (Likert 1-5)
 13  Influencer provides useful info   (Likert 1-5)
 14  Influencer genuinely interested   (Likert 1-5)
 15  More likely to consider (influencer rec) (Likert 1-5)
 16  Demo feels realistic              (Likert 1-5)
 17  Experience seems authentic        (Likert 1-5)
 18  Campaign more relatable           (Likert 1-5)
 19  Influencer explains pros & cons   (Likert 1-5)
 20  Product feels practical           (Likert 1-5)
 21  Content not scripted              (Likert 1-5)
 22  Would consider purchasing         (Likert 1-5)  ← primary DV
 23  Interest increased by campaign    (Likert 1-5)
 24  Understood features better        (Likert 1-5)
 25  Realistic content -> purchase     (Likert 1-5)
 26  Which product to purchase (nominal)
"""

import pandas as pd
import numpy as np

# ── Short aliases for every Likert column ──────────────────────────────────────
LIKERT_ALIAS = {
    "The influencer promoting the product appears knowledgeable about technology products.": "Q_knowledgeable",
    "The influencer seems trustworthy when recommending products.": "Q_trustworthy",
    "I usually trust influencers who regularly review tech gadgets.": "Q_gen_trust",
    "The influencer provides useful information about the product.": "Q_useful_info",
    "The influencer appears genuinely interested in the product rather than just promoting it for sponsorship.": "Q_genuine",
    "I am more likely to consider a product recommended by influencers I follow.": "Q_consider_rec",
    "The product demonstration in the advertisement feels realistic.": "Q_demo_realistic",
    "The influencer's experience with the product seems authentic.": "Q_authentic",
    "The campaign looks more relatable than traditional advertisements.": "Q_relatable",
    "The influencer explains both advantages and limitations of the product": "Q_pros_cons",
    "The campaign makes the product feel practical for everyday use.": "Q_practical",
    "The content does not feel overly scripted or artificial.": "Q_not_scripted",
    "I would consider purchasing one of these earbuds after seeing the campaigns.": "Q_purchase_intent",
    "The influencer promotion increased my interest in the product.": "Q_interest_raised",
    "The campaign helped me understand the product features better.": "Q_understood_features",
    "Realistic influencer content increases my likelihood of purchasing the product.": "Q_realistic_purchase",
}

# ── Feature composition maps ───────────────────────────────────────────────────
TRUST_ITEMS = [
    "Q_trustworthy", "Q_gen_trust", "Q_genuine", "Q_authentic"
]
ENGAGEMENT_ITEMS = [
    "Q_interest_raised", "Q_understood_features", "Q_relatable", "Q_practical"
]
CONTENT_RELEVANCE_ITEMS = [
    "Q_knowledgeable", "Q_useful_info", "Q_demo_realistic",
    "Q_pros_cons", "Q_not_scripted"
]
PURCHASE_INTENT_ITEMS = [
    "Q_purchase_intent", "Q_consider_rec", "Q_realistic_purchase"
]

HOURS_ORDER = {
    "Less than 1 hours": 0.5,
    "1-2 hours": 1.5,
    "2-4 hours": 3.0,
    "More than 4 hours": 5.0,
}

AGE_ORDER = {
    "18-21": 1,
    "22-25": 2,
    "26-30": 3,
    "30 and more": 4,
}


def load_raw(filepath: str) -> pd.DataFrame:
    """Load the raw Excel survey file."""
    df = pd.read_excel(filepath)
    return df


def clean_and_alias(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Likert columns, strip strings, unify encoding."""
    df = df.copy()
    df.rename(columns=LIKERT_ALIAS, inplace=True)

    # Strip whitespace on string columns
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip()

    # Ordinal encode social media hours
    df["hours_numeric"] = df[
        "How many hours do you spend on social media daily?"
    ].map(HOURS_ORDER)

    # Ordinal encode age
    df["age_ordinal"] = df["Age"].map(AGE_ORDER)

    # Binary gender (M/F only; 'Prefer not to say' → NaN for regression)
    df["gender_male"] = (df["Gender"] == "Male").astype(int)
    df.loc[df["Gender"] == "Prefer not to say", "gender_male"] = np.nan

    # One-hot primary platform (collapse rare ones)
    def recode_platform(p):
        if p in ("Instagram", "Youtube"):
            return p
        return "Other"

    df["platform_recoded"] = df[
        "Which social media platform do you use the most?"
    ].apply(recode_platform)
    platform_dummies = pd.get_dummies(
        df["platform_recoded"], prefix="platform", drop_first=False
    )
    df = pd.concat([df, platform_dummies], axis=1)

    # One-hot occupation
    def recode_occ(o):
        if o in ("Student", "Working Profesional"):
            return o
        return "Other"

    df["occ_recoded"] = df["Occupation"].apply(recode_occ)
    occ_dummies = pd.get_dummies(
        df["occ_recoded"], prefix="occ", drop_first=False
    )
    df = pd.concat([df, occ_dummies], axis=1)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create four composite scores.  All Likert items are on 1-5 scale.
    Scores are normalized to 0-1 range for comparability.

    ─────────────────────────────────────────────────────────────────
    TRUST_SCORE
        Measures perceived credibility of the influencer.
        Items: trustworthy | general tech trust | genuine interest | authentic experience
        Formula: mean(Q_trustworthy, Q_gen_trust, Q_genuine, Q_authentic) / 5

    ENGAGEMENT_SCORE
        Measures how much the content captured and held attention.
        Items: interest raised | understood features | relatability | practical feel
        Formula: mean(Q_interest_raised, Q_understood_features, Q_relatable, Q_practical) / 5

    CONTENT_RELEVANCE_INDEX (CRI)
        Measures perceived information quality and realism.
        Items: knowledgeable | useful info | demo realistic | pros & cons | not scripted
        Formula: mean(Q_knowledgeable, Q_useful_info, Q_demo_realistic, Q_pros_cons, Q_not_scripted) / 5

    PURCHASE_INTENT_SCORE (PIS)  ← PRIMARY DEPENDENT VARIABLE
        Measures stated behavioral intention to purchase.
        Items: would consider purchasing | more likely with influencer rec | realistic → purchase
        Formula: mean(Q_purchase_intent, Q_consider_rec, Q_realistic_purchase) / 5
    ─────────────────────────────────────────────────────────────────
    """
    df = df.copy()

    df["Trust_Score"] = df[TRUST_ITEMS].mean(axis=1) / 5
    df["Engagement_Score"] = df[ENGAGEMENT_ITEMS].mean(axis=1) / 5
    df["Content_Relevance_Index"] = df[CONTENT_RELEVANCE_ITEMS].mean(axis=1) / 5
    df["Purchase_Intent_Score"] = df[PURCHASE_INTENT_ITEMS].mean(axis=1) / 5

    # Binary high-intent flag (above median) for logistic regression
    median_pi = df["Purchase_Intent_Score"].median()
    df["High_Intent"] = (df["Purchase_Intent_Score"] >= median_pi).astype(int)

    # Brand preference consistency flag
    # 1 if liked-most == would-purchase (consistent preference signal)
    df["brand_consistency"] = (
        df["Which product did you like the most based on the advertisements?"]
        == df["Which product would you most likely purchase?"]
    ).astype(int)

    return df


def get_model_features(df: pd.DataFrame):
    """
    Return X (feature matrix) and y (target) ready for regression.
    Features chosen: 6 variables — four composite scores + hours + platform_Instagram.
    Drops rows with any NaN in selected columns.
    """
    FEATURES = [
        "Trust_Score",
        "Engagement_Score",
        "Content_Relevance_Index",
        "hours_numeric",
        "platform_Instagram",
        "age_ordinal",
    ]
    TARGET = "Purchase_Intent_Score"

    subset = df[FEATURES + [TARGET]].dropna()
    X = subset[FEATURES]
    y = subset[TARGET]
    return X, y


def get_logistic_features(df: pd.DataFrame):
    """Return X, y for logistic regression (binary High_Intent target)."""
    FEATURES = [
        "Trust_Score",
        "Engagement_Score",
        "Content_Relevance_Index",
        "hours_numeric",
        "platform_Instagram",
        "age_ordinal",
    ]
    TARGET = "High_Intent"
    subset = df[FEATURES + [TARGET]].dropna()
    X = subset[FEATURES]
    y = subset[TARGET]
    return X, y


def get_direct_item_features(df: pd.DataFrame):
    """
    Return X (12 individual Likert items) and y (Purchase Intent Score)
    for the direct-impact model WITHOUT composite scoring.
    This tests whether bypassing aggregation changes conclusions.
    The 3 items that form Purchase Intent Score are excluded from X
    to avoid target leakage.
    """
    # All Likert predictors except the 3 that make up the DV
    DIRECT_FEATURES = [
        "Q_knowledgeable",
        "Q_trustworthy",
        "Q_gen_trust",
        "Q_useful_info",
        "Q_genuine",
        "Q_demo_realistic",
        "Q_authentic",
        "Q_relatable",
        "Q_pros_cons",
        "Q_practical",
        "Q_not_scripted",
        "Q_interest_raised",
    ]
    TARGET = "Purchase_Intent_Score"
    subset = df[DIRECT_FEATURES + [TARGET]].dropna()
    X = subset[DIRECT_FEATURES]
    y = subset[TARGET]
    return X, y


def full_pipeline(filepath: str) -> pd.DataFrame:
    """End-to-end: load → clean → engineer features. Returns analysis-ready df."""
    df = load_raw(filepath)
    df = clean_and_alias(df)
    df = engineer_features(df)
    return df