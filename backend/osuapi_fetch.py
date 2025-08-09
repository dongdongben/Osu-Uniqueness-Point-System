from ossapi import Ossapi
from collections import defaultdict
import csv
import time
import os

client_id = int(os.environ["OSU_CLIENT_ID"])
client_secret = os.environ["OSU_CLIENT_SECRET"]
api = Ossapi(client_id, client_secret)


# Step 1: Get Top 500 Players
UserIDs = []
cursor = None
while len(UserIDs) < 500:
    batch = api.ranking(mode="osu", type="performance", cursor=cursor) if cursor else api.ranking(mode="osu", type="performance")
    UserIDs.extend(entry.user.id for entry in batch.ranking)
    cursor = batch.cursor
    print(f"Collected {len(UserIDs)} user IDs...")

# Step 2: Init tracking structures
total_pp = defaultdict(float)
count = defaultdict(int)
placement_lists = defaultdict(list)
raw_pp_matrix = defaultdict(dict)  # beatmap_id -> {player_id: raw_pp}
beatmap_names = {}
difficulty_name = {}

# Step 3: Fetch and process top 100 plays per user
for i, uid in enumerate(UserIDs):
    print(f"[{i+1}/500] Getting top 100 plays for user {uid}")
    try:
        scores_1 = api.user_scores(uid, type="best", mode="osu", limit=50, offset=0)
        time.sleep(0.2)
        scores_2 = api.user_scores(uid, type="best", mode="osu", limit=50, offset=50)
        scores = scores_1 + scores_2

        for rank, score in enumerate(scores):
            if score.pp is None or score.beatmap is None:
                continue

            beatmap_id = score.beatmap.id
            beatmap_names[beatmap_id] = score.beatmapset.title
            difficulty_name[beatmap_id] = score.beatmap.version
            weighted_pp = score.pp * (0.95 ** rank)

            total_pp[beatmap_id] += weighted_pp
            count[beatmap_id] += 1
            placement_lists[beatmap_id].append(rank + 1)  # placement is 1-indexed
            raw_pp_matrix[beatmap_id][uid] = score.pp

    except Exception as e:
        print(f"Error fetching user {uid}: {e}")
        continue

# Step 4: Write beatmap_profile_stats.csv
with open("backend/data/BeatmapStats.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["BeatmapID", "TotalWeightedPP", "AppearanceCount", "Placements", "Title", "DifficultyName"])
    for beatmap_id in total_pp:
        placements_str = ",".join(str(p) for p in placement_lists[beatmap_id])     
        writer.writerow([
            beatmap_id,
            round(total_pp[beatmap_id], 4),
            count[beatmap_id],
            placements_str,
            beatmap_names.get(beatmap_id, ""),
            difficulty_name.get(beatmap_id, "")
        ])
print(" Written: BeatmapStats.csv")


# Step 5: Write raw_pp_matrix.csv
with open("backend/data/PlayerMatrix.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    header = ["BeatmapID"] + [str(pid) for pid in UserIDs]
    writer.writerow(header)

    for beatmap_id, player_pps in raw_pp_matrix.items():
        row = [beatmap_id]
        for pid in UserIDs:
            val = player_pps.get(pid, "")
            row.append(round(val, 2) if val else "")
        writer.writerow(row)

print(" Written: PlayerMatrix.csv")

# Step 6: Write user ID to username mapping
with open("backend/data/UserID_to_Username.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["UserID", "Username"])
    for uid in UserIDs:
        try:
            user = api.user(uid)
            writer.writerow([uid, user.username])
            print(f"wrote line")
            time.sleep(0.1)  # be gentle with API rate limits
        except Exception as e:
            print(f"Failed to fetch username for {uid}: {e}")

