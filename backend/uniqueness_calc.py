import numpy as np
import pandas as pd
from functools import lru_cache

TOTAL_PLAYERS = 500


def compute_skewness(placements: list[float]) -> float:
    if len(placements) < 3:
        return 0.0
    arr = np.array(placements)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    skewness = np.mean(((arr - mean) / std) ** 3)
    return skewness


def estimate_uniqueness(beatmap_id: int, base_pp: float, appearance_count: int, skewness: float, threshold: int) -> float:
    if appearance_count >= int(threshold):
        factor = skewness * (2 ** (appearance_count / (TOTAL_PLAYERS / 2)) - 0.2)
        return base_pp * (0.93 ** factor)
    else:
        return base_pp * ((1.15 ** (-appearance_count - 11.678393)) + 0.98)

@lru_cache(maxsize=None)
def estimate_uniqueness_cached(beatmap_id, base_pp, appearance_count, skewness, threshold):
    # Don't pass beatmap_stats_df to avoid recursion
    return estimate_uniqueness(beatmap_id, base_pp, appearance_count, skewness, threshold)




def compute_dynamic_threshold(stats: pd.DataFrame) -> int:
    stats = stats.sort_values(by="TotalWeightedPP", ascending=False).reset_index(drop=True)
    n = len(stats)
    x = np.arange(1, n + 1)
    y = stats["TotalWeightedPP"].to_numpy()

    best_error = float('inf')
    best_a = best_b = 0
    steps = 1600
    b_min, b_max = 0.1, 100.0

    for i in range(steps):
        b = b_min + (b_max - b_min) * i / steps
        fit = 1.0 / (x + b)
        a = np.sum(y * fit) / np.sum(fit ** 2)
        error = np.sum((y - a * fit) ** 2)
        if error < best_error:
            best_error = error
            best_a = a
            best_b = b

    y_fit = best_a / (x + best_b)
    min_ac, max_ac = stats["AppearanceCount"].min(), stats["AppearanceCount"].max()
    y_scaled = min_ac + (y_fit - y_fit.min()) / (y_fit.max() - y_fit.min()) * (max_ac - min_ac)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y_scaled - y_scaled.min()) / (y_scaled.max() - y_scaled.min())

    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]

    max_dist = -1
    elbow_index = 0

    for i in range(len(x)):
        px = x_norm[i] - x_norm[0]
        py = y_norm[i] - y_norm[0]
        proj = (px * dx + py * dy) / (dx ** 2 + dy ** 2)
        proj_x = x_norm[0] + proj * dx
        proj_y = y_norm[0] + proj * dy
        dist = np.sqrt((x_norm[i] - proj_x) ** 2 + (y_norm[i] - proj_y) ** 2)
        if dist > max_dist:
            max_dist = dist
            elbow_index = i

    threshold_y = y_scaled[elbow_index]
    return int(round(threshold_y))


def load_beatmap_stats(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_uniqueness_for_player(player_name: str, player_matrix: pd.DataFrame, beatmap_stats: pd.DataFrame) -> pd.DataFrame:
    results = []

    threshold = compute_dynamic_threshold(beatmap_stats)


    for index, row in player_matrix.iterrows():
        beatmap_id = row["BeatmapID"]
        if pd.isna(row[player_name]):
            continue

        base_pp = row[player_name]
        beatmap_row = beatmap_stats[beatmap_stats["BeatmapID"] == beatmap_id]
        if beatmap_row.empty:
            continue

        appearance_count = beatmap_row["AppearanceCount"].values[0]
        appearance_count = int(appearance_count)

        # Drop BeatmapID and get only players who played this map
        player_pps = row.drop("BeatmapID").dropna()

        # Sort by pp descending and assign placement 1..N
        sorted_players = player_pps.sort_values(ascending=False)
        # Get placement string from BeatmapStats and convert to list of ints
        placement_str = beatmap_row["Placements"].values[0]

        if isinstance(placement_str, str) and placement_str.strip():
            placements = [int(p.strip()) for p in placement_str.split(',')]
        else:
            placements = []

        skewness = compute_skewness(placements)
        print(f"Beatmap {beatmap_id} — skewness: {skewness} from placements: {placements}")
        
        # Compute popularity index (always safe to compute)
        popularity_index = 2 ** (appearance_count / (TOTAL_PLAYERS / 2)) - 0.2

        # Compute the full factor (only meaningful if appearance_count >= threshold)
        factor = skewness * popularity_index



        up = estimate_uniqueness(beatmap_id, base_pp, appearance_count, skewness, threshold)
        delta = up - base_pp
        
        
        title = str(beatmap_row.iloc[0]["Title"]) if "Title" in beatmap_row.columns and pd.notna(beatmap_row.iloc[0]["Title"]) else "Unknown"
        difficulty_name = str(beatmap_row.iloc[0]["DifficultyName"]) if "DifficultyName" in beatmap_row.columns and pd.notna(beatmap_row.iloc[0]["DifficultyName"]) else "Unknown"

        full_name = f"{title} [{difficulty_name}]"

        
        results.append({
            "BeatmapID": beatmap_id,
            "BeatmapName": full_name,
            "BasePP": base_pp,  
            "UniquenessPP": up,
            "Delta": delta,
            "AppearanceCount": appearance_count,
            "Skewness": skewness,
            "PopularityIndex": popularity_index,
            "Factor": factor
        })


    df = pd.DataFrame(results)
    df = df.sort_values(by="UniquenessPP", ascending=False).reset_index(drop=True)
    return df


def load_usernames(path="backend/data/UserID_to_Username.csv") -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["UserID"].astype(str), df["Username"]))
username_lookup = load_usernames()


def compute_global_player_rankings(player_matrix: pd.DataFrame, beatmap_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, compute total weighted UniquenessPoints (UP),
    along with average Delta and Factor for ranking display.
    Returns a DataFrame sorted by WeightedUP.
    """
    player_rankings = []
    threshold = int(compute_dynamic_threshold(beatmap_stats))

    beatmap_skewness_lookup = {}

    for _, row in beatmap_stats.iterrows():
        beatmap_id = row["BeatmapID"]
        placement_str = row["Placements"]
        if isinstance(placement_str, str) and placement_str.strip():
            placements = [int(p.strip()) for p in placement_str.split(',')]
            beatmap_skewness_lookup[beatmap_id] = compute_skewness(placements)
        else:
            beatmap_skewness_lookup[beatmap_id] = 0.0

    for i, player in enumerate(player_matrix.columns):

        if player == "BeatmapID":
            continue
        
        print(f"Processing player {i}: {player}")

        player_scores = player_matrix[["BeatmapID", player]].dropna()
        player_scores = player_scores.sort_values(by=player, ascending=False).head(100)

        if player_scores.empty:
            continue

        # Join with BeatmapStats
        merged = player_scores.merge(beatmap_stats, on="BeatmapID", how="inner")

        score_list = []
        for _, row in merged.iterrows():
            beatmap_id = row["BeatmapID"]
            base_pp = row[player]
            appearance_count = row["AppearanceCount"]
            appearance_count = int(appearance_count)


            skewness = beatmap_skewness_lookup.get(beatmap_id, 0.0)

            popularity_index = 2 ** (appearance_count / (TOTAL_PLAYERS / 2)) - 0.2
            factor = skewness * popularity_index
            up = estimate_uniqueness_cached(beatmap_id, base_pp, appearance_count, skewness, threshold)
            delta = up - base_pp

            score_list.append({
                "BasePP": base_pp,
                "UP": up,
                "Delta": delta,
                "Factor": factor
            })

        # Sort by UP and apply weighting
        score_list = sorted(score_list, key=lambda x: x["UP"], reverse=True)
        total_weighted_up = sum(s["UP"] * (0.95 ** i) for i, s in enumerate(score_list[:100]))
        total_weighted_basepp = sum(s["BasePP"] * (0.95 ** i) for i, s in enumerate(score_list[:100]))
        avg_delta = np.mean([s["Delta"] for s in score_list]) if score_list else 0
        avg_factor = np.mean([s["Factor"] for s in score_list]) if score_list else 0

        player_rankings.append({
            "Player": username_lookup.get(str(player), player),
            "PlayerID": str(player),  # <- this line is essential
            "OldPP": total_weighted_basepp,
            "WeightedUP": total_weighted_up,
            "AverageDelta": avg_delta,
            "AverageFactor": avg_factor
        })


    df = pd.DataFrame(player_rankings)
    df = df.sort_values(by="WeightedUP", ascending=False).reset_index(drop=True)
    df.to_csv("backend/data/GlobalRanking.csv", index=False)
    return df
