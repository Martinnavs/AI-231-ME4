import csv, time, numpy as np, torch
from transformers import pipeline
torch.set_num_threads(1)
clf = pipeline("text-classification", model="kalyan-ks/Ammu-1.0-FactCheck-Need-small", device=-1)
rows = list(csv.reader(open("freshqa.csv"))); h = next(i for i, r in enumerate(rows) if r and r[0] == "id")
data = [dict(zip(rows[h], r)) for r in rows[h+1:] if r and r[0].isdigit()]
Q = [d["question"] for d in data]; ft = np.array([d["fact_type"] for d in data])
t = time.perf_counter(); out = [clf(q)[0] for q in Q]; ms = (time.perf_counter() - t) / len(Q) * 1e3
need = np.array([o["label"] == "FACT_CHECK_NEEDED" for o in out])
print("mean latency ms/question (1 thread):", round(ms, 1))
for k in ("never-changing", "slow-changing", "fast-changing"): print(k, "flagged FACT_CHECK_NEEDED:", round(need[ft == k].mean(), 2))
tests = ["What is 34 times 19?", "Tell me a joke about dogs.", "Turn on the lights.", "Play some jazz.", "Write a poem about rain.",
         "How do I poach an egg?", "What is the capital of France?", "Who is the CEO of Nvidia?", "Set a timer for ten minutes.", "Thanks, that's all."]
for q in tests: print(f"{clf(q)[0]['label']:22s} {q}")
