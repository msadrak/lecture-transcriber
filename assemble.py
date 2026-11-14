"""Assemble one lecture's per-segment Gemini transcripts into a Markdown transcript, plus QA flags.

Usage: py -3.12 assemble.py <workdir> <out_md> [--model gemini-3.8-flash-high] [--qa-only]
Reads <workdir>/segments.json, context.json, ag_<model>.jsonl and (optional) pass1.jsonl.
"""
import argparse, json, os, re
from statistics import median

SPK = r"(استاد|دانشجو(?:\s*[۰-۹0-9]+)?)"
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def hms(sec):
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
    t = re.sub(rf"^\s*\**{SPK}\s*:\**", " ", t, flags=re.M)
    return re.findall(r"[\w\u0600-\u06FF]+", t)


def parse_turns(text, seg):
    """-> (turns, bad_ts). turns = list of (abs_seconds or None, speaker or None, paragraph text); relative [mm:ss] -> absolute.
    A stamp that runs backwards by more than 2 s inside a segment (model slip) is dropped, not guessed, and reported in bad_ts."""
    dur = seg["end"] - seg["start"]
    bad_ts = []
    last = [0]

    def to_abs(m):
        t = int(m.group(1)) * 60 + int(m.group(2))
        if t > dur + 3:
            bad_ts.append(f"beyond segment {m.group(0)}")
            t = min(t, dur)
        if t < last[0] - 2:
            bad_ts.append(f"out of order {m.group(0)} after {hms(last[0])[2:]}")
            return None
        last[0] = max(last[0], t)
        return seg["start"] + t

    # join timestamp-only lines with the following line; strip markdown emphasis around labels
    text = re.sub(r"^\s*(\[\d{1,2}:\d{2}\])\s*\n+\s*", r"\1 ", text.replace("\r", ""), flags=re.M)
    text = re.sub(rf"\*\*\s*{SPK}\s*:\s*\*\*", r"\1:", text)
    # "استاد: [00:12] ..." -> "[00:12] استاد: ..." so every turn starts with its timestamp
    text = re.sub(rf"^\s*{SPK}\s*:\s*(\[\d{{1,2}}:\d{{2}}\])\s*", r"\2 \1: ", text, flags=re.M)
    turns = []
    for para in [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]:
        m = re.match(rf"^\[(\d{{1,2}}):(\d{{2}})\]\s*(?:{SPK}\s*:)?\s*(.*)$", para, flags=re.S)
        if m:
            t, spk, body = to_abs(m), m.group(3), m.group(4)
        else:
            m2 = re.match(rf"^{SPK}\s*:\s*(.*)$", para, flags=re.S)
            t, spk, body = (None, m2.group(1), m2.group(2)) if m2 else (None, None, para)
        # inline timestamps inside a paragraph -> absolute
        body = re.sub(r"\[(\d{1,2}):(\d{2})\]", lambda mm: "" if (a := to_abs(mm)) is None else f"[{hms(a)}]", body).strip()
        turns.append((t, spk, body))
    return turns, bad_ts


def repetition_loop(ws, n=4, limit=4):
    grams = {}
    for i in range(len(ws) - n + 1):
        g = tuple(ws[i:i + n])
        grams[g] = grams.get(g, 0) + 1
    worst = max(grams.values(), default=0)
    return worst if worst > limit else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("out_md")
    ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--qa-only", action="store_true")
    a = ap.parse_args()
    wd = a.workdir
    meta = json.load(open(os.path.join(wd, "segments.json"), encoding="utf-8"))
    ctx = json.load(open(os.path.join(wd, "context.json"), encoding="utf-8"))
    segs = meta["segments"]
    ag = load(os.path.join(wd, f"ag_{a.model}.jsonl"))
    p1 = load(os.path.join(wd, "pass1.jsonl"))

    # words per minute of detected speech -> omission check
    rows = []
    for seg in segs:
        r = ag.get(seg["i"])
        w = len(words(r["text"])) if r else 0
        rows.append({"words": w, "wpm": w / (seg["speech"] / 60) if seg["speech"] > 5 else None})
    med = median([r["wpm"] for r in rows if r["wpm"]] or [0])
    flags = []
    for seg, row in zip(segs, rows):
        r = ag.get(seg["i"])
        why = []
        if not r:
            why.append("missing")
        else:
            if r.get("finish") != "STOP":
                why.append(f"finish={r.get('finish')}")
            if row["wpm"] and row["wpm"] < 0.6 * med:
                why.append(f"low density {row['wpm']:.0f} wpm vs median {med:.0f}")
            loop = repetition_loop(words(r["text"]))
            if loop:
                why.append(f"4-gram repeated {loop}x")
            _, bad = parse_turns(r["text"], seg)
            if bad:
                why.append(f"bad timestamps: {bad[:3]}")
            if seg["i"] in p1:
                n1, n2 = len(words(p1[seg["i"]]["text"])), row["words"]
                if n2 < 0.9 * n1:
                    why.append(f"fewer words than gemini-3.5-transcribe draft ({n2} vs {n1})")
        if why:
            flags.append((seg["i"], hms(seg["start"]), why))
    print(f"segments: {len(segs)}, transcribed: {len(ag)}, median density {med:.0f} words/speech-minute, total words {sum(r['words'] for r in rows)}")
    for f in flags:
        print("FLAG seg", f[0], f[1], "; ".join(f[2]))
    if not flags:
        print("no QA flags")
    if a.qa_only:
        return

    out = []
    last_spk = None
    for seg in segs:
        turns, _ = parse_turns(ag[seg["i"]]["text"], seg)
        for t, spk, body in turns:
            if not body:
                continue
            stamp = f"[{hms(t)}] " if t is not None else ""
            if spk and spk != last_spk:
                out.append(f"**{stamp}{spk}:** {body}")
                last_spk = spk
            elif stamp:
                out.append(f"{stamp}{body}")
            else:
                out.append(body)
    header = [
        f"# {ctx['title']}",
        "",
        f"- فایل صوتی: `{os.path.basename(meta['audio'])}` — مدت: {hms(meta['duration']).translate(FA_DIGITS)}",
        "- زمان‌ها به صورت [ساعت:دقیقه:ثانیه] از ابتدای فایل صوتی است.",
        "- «دانشجو ۱»، «دانشجو ۲» فقط دانشجوهای مختلف را در همان بخش از گفتگو از هم جدا می‌کند.",
        "- [نامفهوم]: بخش نامفهوم؛ [همهمه]: صحبت هم‌زمان/نامفهوم چند نفر.",
        "",
        "---",
        "",
    ]
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n\n".join(out) + "\n")
    print("wrote", a.out_md, len(out), "paragraphs")


main()
