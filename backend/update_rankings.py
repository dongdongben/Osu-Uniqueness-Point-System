# backend/update_rankings.py
from backend.uniqueness_calc import compute_global_player_rankings

import pandas as pd

player_matrix = pd.read_csv("backend/data/PlayerMatrix.csv")
beatmap_stats = pd.read_csv("backend/data/BeatmapStats.csv")

df = compute_global_player_rankings(player_matrix, beatmap_stats)
df.to_csv("backend/data/GlobalRanking.csv", index=False)

print("Updated GlobalRanking.csv")
