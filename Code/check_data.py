from ingestion.store import get_feature_store, get_feature_group, get_raw_group

fs = get_feature_store()

raw = get_raw_group(fs).read()
print("raw", raw.shape, raw["time"].min(), raw["time"].max())

feats = get_feature_group(fs).read()
print("features", feats.shape, feats["time"].min(), feats["time"].max())
print("duplicate timestamps:", feats["time"].duplicated().sum())
print("rows without target:", feats["target_aqi"].isna().sum())
print(feats.sort_values("time").tail(3).T)
