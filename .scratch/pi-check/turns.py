import sys, time, json
sys.path.insert(0, "src")
from pathlib import Path
from text_chat.client import AgentClient
tok = Path.home().joinpath(".config/lan_speech_token").read_text().strip()
c = AgentClient("ws://127.0.0.1:8781", tok)
Q = ["According to the GSA TTS handbook, what is a base schedule and where does it live?",
     "In the census population estimates file, what was the estimated United States population in 2024?",
     "What is the weather today?",
     "What is the GSA TTS policy on bringing pets to the office?",
     "Which file did that come from?"]
hist = []
for q in Q:
    t0 = time.time(); first = None; text = []
    for f in c.frames(hist + [{"role": "user", "content": q}]):
        if f.get("text"):
            if first is None: first = time.time() - t0
            text.append(f["text"])
        elif f.get("event") == "tool": text.append("[tool:%s]" % f.get("name"))
        elif f.get("error"): text.append("[error:%s]" % f["error"])
        elif f.get("done"): break
    tot = time.time() - t0
    ans = " ".join(text)
    print(json.dumps({"q": q, "first_s": round(first, 2) if first else None, "total_s": round(tot, 2), "a": ans}, ensure_ascii=False), flush=True)
    hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
