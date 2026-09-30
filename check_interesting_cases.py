import pandas as pd

df = pd.read_csv("results/semantic_margin_metrics.csv")

def margin_drop_table(model):
    clean = df[(df.model==model) & (df.condition=="clean")].set_index("instance_id")
    real = df[(df.model==model) & (df.condition=="real")].set_index("instance_id")
    rows = []
    for iid in clean.index:
        c, r = clean.loc[iid], real.loc[iid]
        rows.append({
            "instance_id": iid, "true_class": c["class"],
            "clean_margin": c["semantic_margin"], "real_margin": r["semantic_margin"],
            "margin_drop": c["semantic_margin"] - r["semantic_margin"],
            "clean_to_real_distance": r["feature_distance"],
            "predicted_class": r["predicted_class"],
        })
    return pd.DataFrame(rows)

for model in ["M2", "M3"]:
    t = margin_drop_table(model)
    med = t["clean_to_real_distance"].median()
    small = t[t["clean_to_real_distance"] <= med]
    top = small.sort_values("margin_drop", ascending=False).head(5)
    print(f"\n=== {model}: BELOW-MEDIAN distance (<= {med:.4f}), ranked by margin drop ===")
    print(top.to_string(index=False))
    top["model"] = model
    top.to_csv(f"results/interesting_cases_{model.lower()}.csv", index=False)