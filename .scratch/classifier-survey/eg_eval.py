import csv, time, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, "/mnt/School/PhD/AI231/AI-231-ME4/src")
torch.set_num_threads(1)
name = "s-nlp/E5-EverGreen-Multilingual-Small"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name).eval()
print("labels:", model.config.id2label)
rows = list(csv.reader(open("freshqa.csv")))
h = next(i for i, r in enumerate(rows) if r and r[0] == "id")
data = [dict(zip(rows[h], r)) for r in rows[h+1:] if r and r[0].isdigit()]
Q = [d["question"] for d in data]
changing = np.array([d["fact_type"] != "never-changing" for d in data])
probs, times = [], []
with torch.no_grad():
    for q in Q[:5]: model(**tok(q, return_tensors="pt"))  # warm up
    for q in Q:
        t = time.perf_counter(); out = model(**tok(q, return_tensors="pt")); times.append(time.perf_counter() - t)
        probs.append(torch.softmax(out.logits, -1)[0].numpy())
probs = np.array(probs)
print("latency single thread, ms: median %.1f p90 %.1f max %.1f" % (np.median(times)*1e3, np.percentile(times, 90)*1e3, max(times)*1e3))
np.save("eg_probs.npy", probs)
for lab in (0, 1):
    # which index means mutable? the one that flags more of the changing questions
    pred = probs.argmax(1) == lab
    tp = (pred & changing).sum(); fp = (pred & ~changing).sum(); fn = (~pred & changing).sum(); tn = (~pred & ~changing).sum()
    print(f"treat label {lab} as 'changing': recall={tp/(tp+fn):.2f} false-alarm={fp/(fp+tn):.2f} precision={tp/max(tp+fp,1):.2f}")
