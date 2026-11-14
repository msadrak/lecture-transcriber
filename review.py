"""Read-only review helpers for one lecture workdir (stdlib only).

  py -3.12 review.py <workdir> show 0,1,57        print those segments from ag_*.jsonl (first 1200 chars)
  py -3.12 review.py <workdir> compare            ag vs pass1 on the sampled segments (similarity, word counts)
  py -3.12 review.py <workdir> pair 20,25         print pass1 and ag text side by side for segments
  py -3.12 review.py <workdir> scan               dates / admin info / Latin-script terms / [نامفهوم] spread
Options: --model gemini-3.8-flash-high
"""
import argparse, difflib, io, json, os, re, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def ts(sec):
    sec = int(round(sec))
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"


def load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            out[r["i"]] = r
    return out


def words(t):
    t = re.sub(r"\[[^\]]*\]|\u200c", " ", t)
    t = re.sub(r"^\s*\**(استاد|دانشجو(?:\s*[۰-۹0-9]+)?)\s*:\**", " ", t, flags=re.M)
    return re.findall(r"[\w\u0600-\u06FF]+", t)


ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("cmd", choices=["show", "compare", "pair", "scan"])
ap.add_argument("ids", nargs="?", default="")
ap.add_argument("--model", default="gemini-3.8-flash-high")
a = ap.parse_args()
wd = a.workdir
segs = json.load(open(os.path.join(wd, "segments.json"), encoding="utf-8"))["segments"]
ag = load(os.path.join(wd, f"ag_{a.model}.jsonl"))
p1 = load(os.path.join(wd, "pass1.jsonl"))
ids = [int(x) for x in a.ids.split(",") if x.strip()]


def head(i):
    s = segs[i]
    return f"=== seg {i} [{ts(s['start'])}-{ts(s['end'])}] len={s['end'] - s['start']:.0f}s speech~{s['speech']}s"


if a.cmd == "show":
    for i in ids:
        print(head(i))
        print(ag[i]["text"][:1200], "\n")

elif a.cmd == "pair":
    for i in ids:
        print(head(i))
        print("PASS1:", (p1[i]["text"] if i in p1 else "(none)")[:700] or "(empty)")
        print("AG   :", ag[i]["text"][:1100], "\n")

elif a.cmd == "compare":
    rows = []
    for i in sorted(p1):
        if i in ag:
            x, y = words(p1[i]["text"]), words(ag[i]["text"])
            rows.append((i, round(difflib.SequenceMatcher(None, x, y, autojunk=False).ratio(), 2), len(x), len(y)))
    sims = sorted(r[1] for r in rows)
    print(f"compared {len(rows)} | similarity median {sims[len(sims) // 2]} min {sims[0]}")
    print(f"words: pass1 {sum(r[2] for r in rows)} vs ag {sum(r[3] for r in rows)}")
    print("ag has fewer words than pass1:", [r for r in rows if r[3] < r[2]])
    print("lowest similarity (seg, sim, pass1 words, ag words):", sorted(rows, key=lambda r: r[1])[:5])
    print("segments with Persian-digit stamps:", sum(1 for r in ag.values() if re.search(r"\[[۰-۹]{2}:[۰-۹]{2}\]", r["text"])))

else:  # scan
    pat = re.compile(r"[^\n]{0,70}(۱۴۰\d|1[34]\d\d|[۰-۹\d]{1,2}/[۰-۹\d]{1,2}|امتحان|نمره|میان‌ترم|میانترم|تکلیف|مهلت|ایمیل|کانال|گروه|ارائه|جلسه‌ی (?:بعد|آینده)|هفته‌ی (?:بعد|آینده)|منبع|مقاله)[^\n]{0,90}")
    for i in sorted(ag):
        for m in pat.finditer(ag[i]["text"]):
            print(f"seg {i} [{ts(segs[i]['start'])}]: ...{m.group(0).strip()}...")
    unclear = sorted(((r["text"].count("[نامفهوم]"), i) for i, r in ag.items() if "[نامفهوم]" in r["text"]), reverse=True)
    print(f"\n[نامفهوم]: {sum(c for c, _ in unclear)} markers in {len(unclear)} segments; top:", [(c, i, ts(segs[i]['start'])) for c, i in unclear[:6]])
    lat = {}
    for r in ag.values():
        for m in re.findall(r"[A-Za-z][A-Za-z0-9 \-']{1,}[A-Za-z0-9]", r["text"]):
            lat[m] = lat.get(m, 0) + 1
    print("Latin-script terms:", sorted(lat.items(), key=lambda kv: -kv[1])[:40])
