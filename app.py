import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from src.data_loader import load_raw_data, build_feature_matrix, pca_variance_analysis
from src.clustering import (
    run_kmeans, get_cluster_assignments, find_similar_courses, find_optimal_k,
    compute_centroid_distances, compute_centroid_linkage,
)
from src.player_performance import (
    load_skill_ratings,
    compute_cluster_profiles,
    score_players_for_cluster,
    label_cluster,
    COURSE_SG_COLS,
    PLAYER_SG_COLS,
)

st.set_page_config(page_title="PGA Course Similarity",
                   page_icon="⛳", layout="wide")


# --- Cached data loading ---
@st.cache_data
def load_data():
    course_df, grass_df = load_raw_data()
    X_scaled, labels, feature_names, scaler, encoder = build_feature_matrix(
        course_df, grass_df)
    skill_df = load_skill_ratings()
    return course_df, grass_df, X_scaled, labels, feature_names, skill_df


@st.cache_data
def compute_optimal_k(X_scaled):
    k_values, inertias, silhouettes, best_k = find_optimal_k(X_scaled)
    return k_values, inertias, silhouettes, best_k


course_df, grass_df, X_scaled, labels, feature_names, skill_df = load_data()
labels_list = labels.tolist()

# --- Header ---
st.title("PGA Golf Course Similarity Analysis")
st.markdown(
    "Explore how PGA Tour courses cluster together based on **{} features** — "
    "course characteristics, strokes-gained demands, and playing conditions. "
    "Find courses that play similarly and discover which players' games best "
    "fit each course type.".format(X_scaled.shape[1])
)

# --- Sidebar ---
st.sidebar.header("Clustering")

k_values, inertias, silhouettes, best_k = compute_optimal_k(X_scaled)

st.sidebar.markdown(
    "**k** is the number of groups courses are sorted into. "
    "Courses within the same group play more similarly to each other "
    "than to courses in other groups."
)
st.sidebar.markdown(f"Optimal k = **{best_k}** (highest silhouette score)")

use_optimal = st.sidebar.toggle("Use optimal k", value=True)

if use_optimal:
    k = best_k
else:
    k = st.sidebar.slider("Number of clusters (k)",
                          min_value=2, max_value=15, value=best_k)

# --- Clustering on full feature space ---
kmeans = run_kmeans(X_scaled, k)
clusters = get_cluster_assignments(kmeans, labels)
cluster_profiles = compute_cluster_profiles(course_df, clusters)

# Build cluster label lookup used across tabs
cluster_label_map = {}
for _, row in cluster_profiles.iterrows():
    cid = int(row["cluster_id"])
    cluster_label_map[cid] = label_cluster(row)

# --- Main tabs ---
tab_cluster, tab_similar, tab_players, tab_data = st.tabs(
    ["Course Clusters", "Find Similar Courses", "Player-Course Fit", "Course Data"]
)

# ---- Tab 1: Cluster Analysis ----
with tab_cluster:

    # --- 1a. Cluster membership cards ---
    st.subheader(f"{k} Course Clusters")
    st.markdown(
        f"Courses are grouped into **{k} clusters** based on {X_scaled.shape[1]} "
        "features including strokes-gained demands, yardage, grass types, soil, "
        "and playing conditions."
    )

    # Render clusters as styled cards in a responsive grid
    ACCENT_COLORS = [
        "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
        "#0891b2", "#be185d", "#4f46e5", "#b45309", "#0d9488",
        "#6d28d9", "#e11d48", "#0284c7", "#15803d", "#ea580c",
    ]

    # Two-column grid of cluster cards
    cols_per_row = 2
    profile_rows = list(cluster_profiles.iterrows())
    for row_start in range(0, len(profile_rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, (_, row) in enumerate(profile_rows[row_start:row_start + cols_per_row]):
            cid = int(row["cluster_id"])
            courses = sorted(clusters[cid])
            clabel = cluster_label_map[cid]
            accent = ACCENT_COLORS[cid % len(ACCENT_COLORS)]
            n = len(courses)

            # Key stats for the card
            yardage = row.get("yardage", 0)
            top_sg = max(
                ("OTT", row.get("ott_sg", 0)),
                ("Approach", row.get("app_sg", 0)),
                ("Short Game", row.get("arg_sg", 0)),
                ("Putting", row.get("putt_sg", 0)),
                key=lambda x: abs(x[1]),
            )
            sg_direction = "+" if top_sg[1] > 0 else "-"

            with cols[col_idx]:
                st.markdown(
                    f"""<div style="
                        border-left: 4px solid {accent};
                        padding: 12px 16px;
                        margin-bottom: 8px;
                        border-radius: 4px;
                        background: rgba(255,255,255,0.03);
                    ">
                        <div style="font-size: 0.8em; color: {accent}; font-weight: 600;
                                    text-transform: uppercase; letter-spacing: 0.05em;">
                            Cluster {cid + 1}
                        </div>
                        <div style="font-size: 1.15em; font-weight: 600; margin: 2px 0 6px 0;">
                            {clabel}
                        </div>
                        <div style="font-size: 0.85em; opacity: 0.7; margin-bottom: 8px;">
                            {n} course{'s' if n != 1 else ''} &nbsp;&bull;&nbsp;
                            Avg {yardage:.0f} yds &nbsp;&bull;&nbsp;
                            Key demand: {sg_direction}{top_sg[0]} ({top_sg[1]:+.3f})
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.expander("View courses", expanded=False):
                    for course in courses:
                        st.markdown(f"- {course}")

    st.divider()

    # --- 1b. Parallel coordinates: what defines each cluster ---
    st.subheader("Cluster Feature Profiles")
    st.markdown(
        "Each line represents a cluster's average across key course features. "
        "Follow the lines to see what makes each cluster distinct."
    )

    parallel_features = [
        "yardage", "adj_driving_distance", "adj_driving_accuracy",
        "ott_sg", "app_sg", "arg_sg", "putt_sg",
        "miss_fw_pen_frac",
    ]
    parallel_display_names = {
        "yardage": "Yardage",
        "adj_driving_distance": "Driving Dist",
        "adj_driving_accuracy": "Driving Acc",
        "ott_sg": "OTT Demand",
        "app_sg": "Approach Demand",
        "arg_sg": "Short Game Demand",
        "putt_sg": "Putting Demand",
        "miss_fw_pen_frac": "Miss FW Penalty",
    }
    display_names_list = list(parallel_display_names.values())

    # Collect raw values per cluster and normalize 0-1
    par_data = []
    for _, row in cluster_profiles.iterrows():
        cid = int(row["cluster_id"])
        vals = {}
        for feat in parallel_features:
            if feat in row.index:
                vals[parallel_display_names[feat]] = row[feat]
        par_data.append(vals)

    par_df = pd.DataFrame(par_data)
    par_min = par_df.min()
    par_max = par_df.max()
    par_norm = (par_df - par_min) / (par_max - par_min)
    par_norm = par_norm.fillna(0.5)

    # Build individual traces per cluster so the legend is discrete
    LINE_COLORS = [
        "#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed",
        "#0891b2", "#be185d", "#4f46e5", "#b45309", "#0d9488",
        "#6d28d9", "#e11d48", "#0284c7", "#15803d", "#ea580c",
    ]

    fig_parallel = go.Figure()
    for i, (_, row) in enumerate(cluster_profiles.iterrows()):
        cid = int(row["cluster_id"])
        y_vals = par_norm.iloc[i].tolist()
        color = LINE_COLORS[cid % len(LINE_COLORS)]
        label = f"C{cid + 1}: {cluster_label_map[cid]}"

        fig_parallel.add_trace(go.Scatter(
            x=display_names_list,
            y=y_vals,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            hovertemplate=(
                "<b>" + label + "</b><br>"
                "%{x}: %{customdata:.3f}<extra></extra>"
            ),
            customdata=par_df.iloc[i].tolist(),
        ))

    fig_parallel.update_layout(
        height=500,
        title="Cluster Profiles Across Key Features (normalized 0-1)",
        xaxis_title="",
        yaxis_title="Normalized Value",
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=30, b=120),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_parallel, width='stretch')

    st.divider()

    # --- 1c. Dendrogram + Heatmap side by side ---
    st.subheader("Cluster Relationships")
    st.markdown(
        "How similar are the clusters to each other? The dendrogram shows a "
        "hierarchy (lower merge = more similar) and the heatmap shows pairwise "
        "distances between cluster centroids."
    )

    col_dendro, col_heat = st.columns(2)

    with col_dendro:
        Z = compute_centroid_linkage(kmeans)
        dendro_labels = [
            f"C{int(row['cluster_id']) + 1}"
            for _, row in cluster_profiles.iterrows()
        ]
        fig_dendro = ff.create_dendrogram(
            kmeans.cluster_centers_,
            labels=dendro_labels,
            linkagefun=lambda x: Z,
        )
        fig_dendro.update_layout(
            height=450,
            xaxis_title="Cluster",
            yaxis_title="Distance (Ward)",
            plot_bgcolor="white",
            title="Cluster Dendrogram",
        )
        st.plotly_chart(fig_dendro, width='stretch')

    with col_heat:
        dist_matrix = compute_centroid_distances(kmeans)
        heatmap_labels = [f"C{cid + 1}" for cid in sorted(clusters.keys())]

        fig_heat = px.imshow(
            dist_matrix,
            x=heatmap_labels,
            y=heatmap_labels,
            color_continuous_scale="Blues_r",
            labels={"color": "Distance"},
            title="Inter-Cluster Distance",
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, width='stretch')

    # --- 1d. Optimal k analysis ---
    with st.expander("Optimal k Analysis", expanded=False):
        st.markdown(
            f"Clustering is performed on the **full {X_scaled.shape[1]}-dimensional "
            "feature space** using K-Means. The optimal k is selected by maximizing "
            "the silhouette score."
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_elbow = px.line(
                x=k_values, y=inertias,
                labels={"x": "k", "y": "Inertia"},
                title="Elbow Method",
                markers=True,
            )
            fig_elbow.add_vline(x=best_k, line_dash="dash", line_color="red",
                                annotation_text=f"Best k={best_k}")
            st.plotly_chart(fig_elbow, width='stretch')
        with col2:
            fig_sil = px.line(
                x=k_values, y=silhouettes,
                labels={"x": "k", "y": "Silhouette Score"},
                title="Silhouette Score",
                markers=True,
            )
            fig_sil.add_vline(x=best_k, line_dash="dash", line_color="red",
                              annotation_text=f"Best k={best_k}")
            st.plotly_chart(fig_sil, width='stretch')

        st.success(
            f"Optimal **k = {best_k}** — silhouette score: **{max(silhouettes):.3f}**")

# ---- Tab 2: Similar Course Finder ----
with tab_similar:
    st.subheader("Find Similar Courses")
    st.markdown(
        "Similarity is computed using nearest-neighbor distance across all "
        f"**{X_scaled.shape[1]} features**. Select a course to see its closest matches "
        "and how they compare on key characteristics."
    )
    selected_course = st.selectbox("Select a course", sorted(labels_list))

    if selected_course:
        # Pull a generous set of neighbors; the slider below controls how many to display
        all_similar = find_similar_courses(
            selected_course, X_scaled, labels, n_neighbors=20)

        st.markdown(f"**Courses most similar to {selected_course}:**")

        sim_df = pd.DataFrame(all_similar, columns=["Course", "Distance"])
        sim_df.index = range(1, len(sim_df) + 1)
        sim_df.index.name = "Rank"
        sim_df["Similarity"] = 1 / (1 + sim_df["Distance"])

        # Add cluster assignment for context
        course_to_cluster = {}
        for cid, courses in clusters.items():
            for c in courses:
                course_to_cluster[c] = cid
        sim_df["Cluster"] = sim_df["Course"].map(
            lambda c: f"C{course_to_cluster.get(c, 0) + 1}: {cluster_label_map.get(course_to_cluster.get(c, 0), '')}"
        )
        selected_cluster = course_to_cluster.get(selected_course, 0)
        st.caption(
            f"**{selected_course}** belongs to "
            f"**Cluster {selected_cluster + 1}: {cluster_label_map.get(selected_cluster, '')}**"
        )

        n_to_show = st.slider("Number of results",
                              min_value=3, max_value=20, value=8)
        st.dataframe(
            sim_df[["Course", "Similarity", "Cluster"]].head(n_to_show),
            width='stretch',
        )

        # Feature comparison: selected course vs. similar courses
        st.subheader("Feature Comparison")
        compare_features = [
            "yardage", "adj_driving_distance", "adj_driving_accuracy",
            "ott_sg", "app_sg", "arg_sg", "putt_sg",
        ]
        compare_names = ["Yardage", "Driving Dist", "Driving Acc",
                         "OTT Demand", "Approach Demand", "Short Game Demand", "Putting Demand"]

        compare_courses = [selected_course] + [c for c, _ in all_similar[:5]]
        compare_rows = []
        for c in compare_courses:
            row_data = course_df[course_df["course"] == c]
            if not row_data.empty:
                r = row_data.iloc[0]
                entry = {"Course": c}
                for feat, name in zip(compare_features, compare_names):
                    if feat in r.index:
                        val = r[feat]
                        if val == "undefined":
                            val = np.nan
                        entry[name] = float(
                            val) if not pd.isna(val) else np.nan
                compare_rows.append(entry)

        if compare_rows:
            compare_df = pd.DataFrame(compare_rows).set_index("Course")

            # Normalize for radar overlay
            norm_df = (compare_df - compare_df.min()) / \
                (compare_df.max() - compare_df.min())
            norm_df = norm_df.fillna(0)

            fig_compare = go.Figure()
            for i, course_name in enumerate(norm_df.index):
                vals = norm_df.loc[course_name].tolist(
                ) + [norm_df.loc[course_name].tolist()[0]]
                fig_compare.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=compare_names + [compare_names[0]],
                    name=course_name,
                    fill="toself" if i == 0 else "none",
                    opacity=0.7 if i == 0 else 0.4,
                    line=dict(width=3 if i == 0 else 1.5),
                    hovertemplate=f"<b>{course_name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
                ))
            fig_compare.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"{selected_course} vs. Top {min(5, len(all_similar))} Similar Courses",
                height=500,
                hoverlabel=dict(bgcolor="#1e1e1e",
                                font_color="white", font_size=13),
            )
            st.plotly_chart(fig_compare, width='stretch')

            st.caption(
                "Radar values are normalized 0–1 across the compared courses for visual comparison.")

# ---- Tab 3: Player-Course Fit ----
with tab_players:
    st.subheader("Player-Course Fit Analysis")
    st.markdown(
        "Each course cluster has a distinct **strokes-gained demand profile** — how much "
        "putting, approach play, off-the-tee, and short game matter there. By matching "
        "player skill profiles against these demands, we can predict which players' games "
        "best fit each course type."
    )

    # --- Cluster profile overview ---
    st.subheader("Cluster SG Demand Profiles")

    profile_display = []
    for _, row in cluster_profiles.iterrows():
        profile_display.append({
            "Cluster": f"Cluster {int(row['cluster_id']) + 1}",
            "Label": label_cluster(row),
            "Courses": int(row["n_courses"]),
            "Avg Yardage": f"{row['yardage']:.0f}",
            "OTT Demand": f"{row['ott_sg']:+.3f}",
            "Approach Demand": f"{row['app_sg']:+.3f}",
            "Short Game Demand": f"{row['arg_sg']:+.3f}",
            "Putting Demand": f"{row['putt_sg']:+.3f}",
        })
    st.dataframe(pd.DataFrame(profile_display),
                 width='stretch', hide_index=True)

    # Radar chart of cluster profiles
    sg_labels = ["Off-the-Tee", "Approach", "Short Game", "Putting"]
    fig_radar = go.Figure()
    for _, row in cluster_profiles.iterrows():
        values = [row[c] for c in COURSE_SG_COLS]
        values.append(values[0])
        cname = f"C{int(row['cluster_id']) + 1}: {cluster_label_map[int(row['cluster_id'])]}"
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=sg_labels + [sg_labels[0]],
            name=cname,
            fill="toself",
            opacity=0.4,
            hovertemplate=f"<b>{cname}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Cluster SG Demand Profiles (Radar)",
        height=500,
        hoverlabel=dict(bgcolor="#1e1e1e", font_color="white", font_size=13),
    )
    st.plotly_chart(fig_radar, width='stretch')

    st.divider()

    # --- Player fit for a selected cluster ---
    st.subheader("Best-Fit Players by Cluster")

    col_select, col_count = st.columns([2, 1])
    with col_select:
        cluster_options = {
            f"Cluster {int(row['cluster_id']) + 1} — {cluster_label_map[int(row['cluster_id'])]}": int(row['cluster_id'])
            for _, row in cluster_profiles.iterrows()
        }
        selected_cluster_label = st.selectbox(
            "Select a cluster", list(cluster_options.keys()))
        selected_cluster_id = cluster_options[selected_cluster_label]
    with col_count:
        top_n = st.slider("Players to show", min_value=5,
                          max_value=50, value=20)

    profile_row = cluster_profiles[cluster_profiles["cluster_id"]
                                   == selected_cluster_id].iloc[0]
    scored = score_players_for_cluster(skill_df, profile_row)

    cluster_courses = clusters[selected_cluster_id]
    st.caption(
        f"**Courses in this cluster:** {', '.join(sorted(cluster_courses))}")

    display_cols = ["player_name", "sg_total", "sg_ott",
                    "sg_app", "sg_arg", "sg_putt", "fit_score", "predicted_sg"]
    top_players = scored[display_cols].head(top_n).copy()
    top_players.columns = ["Player", "Baseline SG", "SG OTT", "SG Approach",
                           "SG Short Game", "SG Putting", "Course Fit", "Predicted SG"]

    st.dataframe(
        top_players.style.format({
            "Baseline SG": "{:+.3f}",
            "SG OTT": "{:+.3f}",
            "SG Approach": "{:+.3f}",
            "SG Short Game": "{:+.3f}",
            "SG Putting": "{:+.3f}",
            "Course Fit": "{:+.4f}",
            "Predicted SG": "{:+.3f}",
        }).background_gradient(subset=["Course Fit"], cmap="RdYlGn"),
        width='stretch',
    )

    # Bar chart: fit score breakdown for top 10
    st.subheader("Top 10 — Skill vs. Course Fit Breakdown")
    top10 = scored.head(10).copy()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=top10["player_name"],
        x=top10["sg_total"],
        name="Baseline Skill (SG Total)",
        orientation="h",
        marker_color="#3498db",
    ))
    fig_bar.add_trace(go.Bar(
        y=top10["player_name"],
        x=top10["fit_score"],
        name="Course Fit Bonus",
        orientation="h",
        marker_color="#2ecc71",
    ))
    fig_bar.update_layout(
        barmode="stack",
        title="Predicted Strokes Gained = Baseline Skill + Course Fit",
        xaxis_title="Strokes Gained",
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    st.plotly_chart(fig_bar, width='stretch')

    st.divider()

    # --- Look up a specific player ---
    st.subheader("Player Lookup")
    player_names = sorted(skill_df["player_name"].tolist())
    selected_player = st.selectbox("Search for a player", player_names)

    if selected_player:
        player_row = skill_df[skill_df["player_name"]
                              == selected_player].iloc[0]

        col_profile, col_radar = st.columns(2)
        with col_profile:
            st.markdown(f"**{selected_player}**")
            st.metric("Overall SG", f"{player_row['sg_total']:+.3f}")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.metric("SG Off-the-Tee", f"{player_row['sg_ott']:+.3f}")
                st.metric("SG Approach", f"{player_row['sg_app']:+.3f}")
            with pcol2:
                st.metric("SG Short Game", f"{player_row['sg_arg']:+.3f}")
                st.metric("SG Putting", f"{player_row['sg_putt']:+.3f}")

        with col_radar:
            player_vals = [player_row[c]
                           for c in PLAYER_SG_COLS] + [player_row[PLAYER_SG_COLS[0]]]
            fig_player_radar = go.Figure()
            fig_player_radar.add_trace(go.Scatterpolar(
                r=player_vals,
                theta=sg_labels + [sg_labels[0]],
                fill="toself",
                name=selected_player,
                hovertemplate=f"<b>{selected_player}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
            ))
            fig_player_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=f"{selected_player} — SG Profile",
                height=350,
                hoverlabel=dict(bgcolor="#1e1e1e",
                                font_color="white", font_size=13),
            )
            st.plotly_chart(fig_player_radar, width='stretch')

        # Player's fit across all clusters
        st.markdown(f"**{selected_player}'s fit across all course clusters:**")
        fit_rows = []
        for _, profile in cluster_profiles.iterrows():
            cid = int(profile["cluster_id"])
            scored_all = score_players_for_cluster(skill_df, profile)
            player_scored = scored_all[scored_all["player_name"]
                                       == selected_player]
            if not player_scored.empty:
                ps = player_scored.iloc[0]
                rank = player_scored.index[0]
                fit_rows.append({
                    "Cluster": f"Cluster {cid + 1}",
                    "Label": cluster_label_map[cid],
                    "Courses": ", ".join(sorted(clusters[cid])),
                    "Fit Score": ps["fit_score"],
                    "Predicted SG": ps["predicted_sg"],
                    "Rank": rank,
                })

        fit_df = pd.DataFrame(fit_rows).sort_values(
            "Predicted SG", ascending=False)
        st.dataframe(
            fit_df.style.format({
                "Fit Score": "{:+.4f}",
                "Predicted SG": "{:+.3f}",
            }).background_gradient(subset=["Fit Score"], cmap="RdYlGn"),
            width='stretch',
            hide_index=True,
        )

# ---- Tab 4: Raw Data Explorer ----
with tab_data:
    st.subheader("Course Performance Data")
    st.dataframe(course_df, width='stretch', height=400)

    st.subheader("Course Characteristics")
    st.dataframe(grass_df, width='stretch', height=400)

    st.subheader("Player Skill Ratings")
    st.dataframe(
        skill_df.sort_values(
            "sg_total", ascending=False).reset_index(drop=True),
        width='stretch',
        height=400,
    )

    st.subheader("Dimensionality Info")
    pca_full, cumulative = pca_variance_analysis(X_scaled)
    n_for_85 = int(np.argmax(cumulative >= 0.85) + 1)
    n_for_95 = int(np.argmax(cumulative >= 0.95) + 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Features", X_scaled.shape[1])
    col2.metric("Components for 85% Variance", n_for_85)
    col3.metric("Components for 95% Variance", n_for_95)

    fig_var = px.line(
        x=list(range(1, len(cumulative) + 1)),
        y=cumulative,
        labels={"x": "Number of Components",
                "y": "Cumulative Variance Explained"},
        title="PCA Cumulative Variance Explained",
        markers=True,
    )
    fig_var.add_hline(y=0.85, line_dash="dash",
                      line_color="orange", annotation_text="85%")
    fig_var.add_hline(y=0.95, line_dash="dash",
                      line_color="red", annotation_text="95%")
    fig_var.update_layout(height=400)
    st.plotly_chart(fig_var, width='stretch')

    st.caption(
        "All clustering and similarity analysis uses the full "
        f"{X_scaled.shape[1]}-dimensional feature space."
    )
