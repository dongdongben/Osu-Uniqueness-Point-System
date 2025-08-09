from flask import Flask, render_template
import pandas as pd
from flask import send_file
from backend.uniqueness_calc import (
    compute_uniqueness_for_player,
    compute_global_player_rankings,
    load_usernames,
)




app = Flask(__name__)

# === Run once at startup ===
username_lookup = load_usernames()
player_matrix = pd.read_csv("backend/data/PlayerMatrix.csv")
beatmap_stats = pd.read_csv("backend/data/BeatmapStats.csv", encoding="ISO-8859-1")

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/ranking")
def global_ranking():
    df = pd.read_csv("backend/data/GlobalRanking.csv")
    return render_template("ranking.html", rankings=df.to_dict(orient="records"))

@app.route("/player/<user_id>")
def player_profile(user_id):
    if user_id not in player_matrix.columns:
        return f"<h2>Player '{user_id}' not found</h2>", 404
    username = username_lookup.get(str(user_id), f"User {user_id}")
    df = compute_uniqueness_for_player(str(user_id), player_matrix, beatmap_stats)
    return render_template("leaderboard.html", name=username, scores=df.to_dict(orient="records"))

@app.route("/data/beatmapstats.csv")
def beatmapstats_csv():
    # Serves the same CSV your backend is already writing/using
    return send_file("backend/data/BeatmapStats.csv", mimetype="text/csv")

@app.route("/data/weightedpp_all.json")
def weightedpp_all():
    import pandas as pd
    from flask import jsonify

    df = pd.read_csv("backend/data/BeatmapStats.csv", encoding="ISO-8859-1")

    def norm(s): return s.strip().lower().replace(" ", "")

    col_pp    = next((c for c in df.columns if norm(c) == "totalweightedpp"), None)
    col_title = next((c for c in df.columns if norm(c) == "title"), None)
    col_diff  = next((c for c in df.columns if norm(c) == "difficultyname"), None)

    if col_pp is None:
        return jsonify({"x": [], "y": [], "error": "TotalWeightedPP column not found"}), 500

    # Beatmap display name (Title [Difficulty]) with fallbacks
    if col_title and col_diff:
        names = (df[col_title].fillna("Unknown").astype(str) + 
                 " [" + df[col_diff].fillna("Unknown").astype(str) + "]")
    elif col_title:
        names = df[col_title].fillna("Unknown").astype(str)
    else:
        names = df.get("BeatmapID", pd.Series(range(len(df)))).astype(str)

    # Clean numeric pp
    pp = (df[col_pp].astype(str)
                 .str.replace(r"[^0-9eE\.\-+]", "", regex=True)
                 .replace("", pd.NA)
                 .astype(float))

    out = pd.DataFrame({"name": names, "pp": pp}).dropna()
    out = out.sort_values("pp", ascending=False)  # full, no limit

    return jsonify({
        "x": out["name"].tolist(),  # names for hover
        "y": out["pp"].tolist()     # all pp values, sorted desc
    })

@app.route("/data/appearance_aligned.json")
def appearance_aligned():
    import pandas as pd
    from flask import jsonify

    df = pd.read_csv("backend/data/BeatmapStats.csv", encoding="ISO-8859-1")

    def norm(s): return s.strip().lower().replace(" ", "")

    col_pp    = next((c for c in df.columns if norm(c) == "totalweightedpp"), None)
    col_app   = next((c for c in df.columns if norm(c) == "appearancecount"), None)
    col_title = next((c for c in df.columns if norm(c) == "title"), None)
    col_diff  = next((c for c in df.columns if norm(c) == "difficultyname"), None)

    if col_pp is None or col_app is None:
        return jsonify({"x": [], "y": [], "error": "Required columns not found"}), 500

    # Beatmap display name
    if col_title and col_diff:
        names = (df[col_title].fillna("Unknown").astype(str) +
                 " [" + df[col_diff].fillna("Unknown").astype(str) + "]")
    elif col_title:
        names = df[col_title].fillna("Unknown").astype(str)
    else:
        names = df.get("BeatmapID", pd.Series(range(len(df)))).astype(str)

    # Clean numeric columns
    pp = (df[col_pp].astype(str)
                  .str.replace(r"[^0-9eE\.\-+]", "", regex=True)
                  .replace("", pd.NA).astype(float))
    app = (df[col_app].astype(str)
                    .str.replace(r"[^0-9\.\-+]", "", regex=True)
                    .replace("", pd.NA).astype(float))

    out = pd.DataFrame({"name": names, "pp": pp, "app": app}).dropna()

    # IMPORTANT: sort by pp DESC and keep that order for appearance
    out = out.sort_values("pp", ascending=False)

    return jsonify({
        "names": out["name"].tolist(),   # for hover
        "pp":    out["pp"].tolist(),     # same order
        "app":   out["app"].tolist()     # appearance counts aligned to pp order
    })

@app.route("/data/fit_scaled_elbow.json")
def fit_scaled_elbow():
    import numpy as np
    import pandas as pd
    from flask import jsonify

    # --- load & clean ---
    df = pd.read_csv("backend/data/BeatmapStats.csv", encoding="ISO-8859-1")

    def norm(s): return s.strip().lower().replace(" ", "")

    col_pp    = next((c for c in df.columns if norm(c) == "totalweightedpp"), None)
    col_app   = next((c for c in df.columns if norm(c) == "appearancecount"), None)
    col_title = next((c for c in df.columns if norm(c) == "title"), None)
    col_diff  = next((c for c in df.columns if norm(c) == "difficultyname"), None)

    if col_pp is None or col_app is None:
        return jsonify({"error": "Required columns not found"}), 500

    # Beatmap name for hover
    if col_title and col_diff:
        names = (df[col_title].fillna("Unknown").astype(str) +
                 " [" + df[col_diff].fillna("Unknown").astype(str) + "]")
    elif col_title:
        names = df[col_title].fillna("Unknown").astype(str)
    else:
        names = df.get("BeatmapID", pd.Series(range(len(df)))).astype(str)

    # Numeric cleaning
    def clean_num(series):
        return (series.astype(str)
                      .str.replace(r"[^0-9eE\.\-+]", "", regex=True)
                      .replace("", pd.NA).astype(float))

    pp  = clean_num(df[col_pp])
    app = clean_num(df[col_app])

    # Drop NaNs & align
    out = pd.DataFrame({"name": names, "pp": pp, "app": app}).dropna()

    # --- sort by pp DESC; keep this order for everything ---
    out = out.sort_values("pp", ascending=False).reset_index(drop=True)

    x = np.arange(1, len(out) + 1, dtype=float)   # 1..N
    y = out["pp"].to_numpy(dtype=float)           # Weighted PP
    app_vals = out["app"].to_numpy(dtype=float)   # Appearance Count
    names_lst = out["name"].tolist()

    # --- fit y ≈ a/(x + b) via 1D search over b (fast, robust, no SciPy needed) ---
    # For each b, least-squares a = sum(y * f) / sum(f^2) where f = 1/(x + b)
    steps = 2000
    b_min, b_max = 0.1, 100.0
    best = {"err": np.inf, "a": None, "b": None}

    # Precompute once
    for i in range(steps):
        b = b_min + (b_max - b_min) * i / (steps - 1)
        f = 1.0 / (x + b)
        denom = np.sum(f * f)
        if denom == 0:
            continue
        a = np.sum(y * f) / denom
        yhat = a * f
        err = np.sum((y - yhat) ** 2)
        if err < best["err"]:
            best = {"err": err, "a": a, "b": b}

    a, b = float(best["a"]), float(best["b"])
    y_fit = a / (x + b)

    # --- scale y_fit to the range of appearance counts ---
    # min-max scale y_fit to [min(app), max(app)]
    y_min, y_max = y_fit.min(), y_fit.max()
    if y_max == y_min:
        y_fit_scaled = np.full_like(y_fit, app_vals.mean())
    else:
        app_min, app_max = float(app_vals.min()), float(app_vals.max())
        y_fit_scaled = app_min + (y_fit - y_min) * (app_max - app_min) / (y_max - y_min)

    # --- elbow by max-distance-to-chord on the scaled curve ---
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y_fit_scaled - y_fit_scaled.min()) / (y_fit_scaled.max() - y_fit_scaled.min())

    start_x, start_y = x_norm[0],  y_norm[0]
    end_x,   end_y   = x_norm[-1], y_norm[-1]
    dx, dy = (end_x - start_x), (end_y - start_y)
    denom = (dx*dx + dy*dy)

    # projection of each point onto the start->end line, distance to the line
    px = x_norm - start_x
    py = y_norm - start_y
    proj = (px * dx + py * dy) / (denom if denom != 0 else 1.0)
    proj_x = start_x + proj * dx
    proj_y = start_y + proj * dy
    dists = np.sqrt((x_norm - proj_x)**2 + (y_norm - proj_y)**2)

    elbow_idx = int(np.argmax(dists))           # zero-based index
    elbow_x   = int(x[elbow_idx])               # 1-based position
    elbow_y   = float(y_fit_scaled[elbow_idx])  # scaled value at elbow

    return jsonify({
        "x": x.astype(int).tolist(),
        "names": names_lst,
        "pp": y.tolist(),                 # the fitted target (for reference if you want)
        "app": app_vals.tolist(),         # actual appearance counts (pp-sorted)
        "fit_scaled": y_fit_scaled.tolist(),
        "fit_params": {"a": a, "b": b},
        "elbow": {"index0": elbow_idx, "x": elbow_x, "y": elbow_y}
    })






if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
