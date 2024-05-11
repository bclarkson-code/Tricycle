import humanize
import pandas as pd

df = pd.read_csv("memory.log")
df["memory_diff"] = df["used_bytes"] - df["used_bytes"].shift()
df["time_diff"] = df["timestamp"] - df["timestamp"].shift()
df = df.dropna()

print(df.iloc[:49])
grouped = df.groupby("stage").agg({"memory_diff": "mean", "time_diff": "mean"})
grouped = grouped.sort_values(by="memory_diff", ascending=False)
grouped["memory_diff"] = grouped["memory_diff"].apply(humanize.naturalsize)
grouped["time_diff"] = (grouped["time_diff"] * 1e6).astype(int)


print(grouped)

# df["memory_diff_human"] = df["memory_diff"].apply(humanize.naturalsize)
# print(df[["stage", "memory_diff_human", "time_diff_μs"]])
