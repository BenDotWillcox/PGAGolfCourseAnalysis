"""Map player skill profiles to course clusters to predict player-course fit.

The idea: each course has SG "demands" (how much putting, approach, OTT, ARG
matter there). Each player has SG "strengths". By computing the dot product
of a player's SG profile with a cluster's average SG demand profile, we get
a course-fit score — how well the player's game matches what the cluster rewards.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# SG categories that exist in BOTH the course data and the player skill ratings
# Course columns describe how much each SG category separates players at that course
# Player columns describe each player's skill in that category
COURSE_SG_COLS = ["ott_sg", "app_sg", "arg_sg", "putt_sg"]
PLAYER_SG_COLS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

# Additional course demand columns for richer fit analysis
COURSE_DETAIL_COLS = [
    "adj_driving_distance", "adj_driving_accuracy",
    "less_150_sg", "greater_150_sg",
    "arg_fairway_sg", "arg_rough_sg", "arg_bunker_sg",
    "less_5_ft_sg", "greater_5_less_15_sg", "greater_15_sg",
]


def load_skill_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "skill_ratings.csv")


def load_rankings() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "rankings.csv")


def compute_cluster_profiles(course_df: pd.DataFrame, cluster_assignments: dict) -> pd.DataFrame:
    """Compute the average SG demand profile for each cluster.

    Returns a DataFrame with one row per cluster and columns for each SG category,
    plus descriptive stats (avg yardage, number of courses, course names).
    """
    rows = []
    for cluster_id, courses in cluster_assignments.items():
        cluster_courses = course_df[course_df["course"].isin(courses)]
        if cluster_courses.empty:
            continue

        row = {"cluster_id": cluster_id, "n_courses": len(courses)}
        row["courses"] = ", ".join(sorted(courses))

        # Average SG demands
        for col in COURSE_SG_COLS:
            if col in cluster_courses.columns:
                row[col] = cluster_courses[col].mean()

        # Average physical characteristics
        for col in ["yardage", "adj_driving_distance", "adj_driving_accuracy",
                     "fw_width", "miss_fw_pen_frac"]:
            if col in cluster_courses.columns:
                row[col] = cluster_courses[col].mean()

        # Detail SG columns
        for col in COURSE_DETAIL_COLS:
            if col in cluster_courses.columns:
                row[col] = cluster_courses[col].mean()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def score_players_for_cluster(
    skill_df: pd.DataFrame,
    cluster_profile: pd.Series,
) -> pd.DataFrame:
    """Score every player's fit for a single cluster.

    Fit score = sum(player_sg_i * cluster_demand_i) for each SG category.
    A positive fit score means the player's strengths align with what the
    cluster rewards. Higher is better.
    """
    fit_scores = np.zeros(len(skill_df))

    for p_col, c_col in zip(PLAYER_SG_COLS, COURSE_SG_COLS):
        if p_col in skill_df.columns and c_col in cluster_profile.index:
            demand = cluster_profile[c_col]
            fit_scores += skill_df[p_col].values * demand

    result = skill_df[["dg_id", "player_name"] + PLAYER_SG_COLS].copy()
    result["fit_score"] = fit_scores
    result["sg_total"] = skill_df["sg_total"]
    # Overall predicted performance = baseline skill + course fit bonus
    result["predicted_sg"] = result["sg_total"] + result["fit_score"]
    result = result.sort_values("predicted_sg", ascending=False).reset_index(drop=True)
    result.index = result.index + 1
    result.index.name = "rank"
    return result


def score_player_for_course(
    player_row: pd.Series,
    course_row: pd.Series,
) -> float:
    """Score a single player's fit for a single course."""
    fit = 0.0
    for p_col, c_col in zip(PLAYER_SG_COLS, COURSE_SG_COLS):
        fit += player_row[p_col] * course_row[c_col]
    return fit


def score_all_players_all_clusters(
    skill_df: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Build a matrix of player x cluster fit scores."""
    records = []
    for _, profile in cluster_profiles.iterrows():
        cluster_id = profile["cluster_id"]
        scored = score_players_for_cluster(skill_df, profile)
        scored["cluster_id"] = cluster_id
        records.append(scored)
    return pd.concat(records, ignore_index=True)


def label_cluster(profile: pd.Series) -> str:
    """Generate a short descriptive label for a cluster based on its dominant demands."""
    labels = []

    # Check which SG category has highest absolute demand
    sg_demands = {col: profile.get(col, 0) for col in COURSE_SG_COLS}
    top_demand = max(sg_demands, key=lambda k: abs(sg_demands[k]))
    demand_map = {
        "ott_sg": "Off-the-Tee",
        "app_sg": "Approach",
        "arg_sg": "Short Game",
        "putt_sg": "Putting",
    }
    if abs(sg_demands[top_demand]) > 0.01:
        sign = "+" if sg_demands[top_demand] > 0 else "-"
        labels.append(f"{sign}{demand_map[top_demand]}")

    # Yardage
    yardage = profile.get("yardage", 0)
    if yardage > 7300:
        labels.append("Long")
    elif yardage < 7050:
        labels.append("Short")

    # Accuracy
    acc = profile.get("adj_driving_accuracy", 0)
    if acc < 0.60:
        labels.append("Wide Open")
    elif acc > 0.68:
        labels.append("Tight")

    return " | ".join(labels) if labels else "Balanced"
