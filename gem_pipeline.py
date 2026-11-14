"""Gemini transcription of one lecture's pre-cut segments (stdlib only, resumable).

A lecture lives in a workdir made by segment.py (segments.json + seg/*.flac) plus a hand-written
context.json: {"title": <Persian transcript title>, "course": <course description for the prompt>,
               "terms_rule": <spelling guidance: names, books, foreign-term script>}.

pass1: gemini-3.5-transcribe via the Gemini API key -> plain verbatim draft (free tier: 25 requests/day;
       stops at the daily quota)                                            (out: <workdir>/pass1.jsonl)
ag:    a Gemini model via the Oh My Pi Antigravity OAuth accounts -> verbatim transcript with speaker
       labels, punctuation, [mm:ss] segment timestamps                      (out: <workdir>/ag_<model>.jsonl)

Usage:
  py -3.12 gem_pipeline.py ag <workdir> [--model gemini-3.8-flash-high] [--workers 3] [--only 0,1,2]
  py -3.12 gem_pipeline.py pass1 <workdir> [--workers 4] [--only ...]
Secrets (GEMINI_API_KEY in ~/.omp/agent/.env, OAuth tokens in ~/.omp/agent/agent.db) are never printed.
"""
import argparse, base64, concurrent.futures as cf, difflib, itertools, json, os, re, sqlite3, subprocess, sys, threading, time, urllib.error, urllib.request, uuid

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OMP = os.path.expanduser("~/.omp/agent")
AG_BASE = "https://daily-cloudcode-pa.googleapis.com"
AG_UA = "antigravity/hub/2.19.1 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"

PROMPT_TEMPLATE = """The attached audio is one segment of a Persian university lecture (course: {course}). The lecturer speaks most of the time; students occasionally ask questions or answer.

Produce a complete, verbatim Persian transcript of this audio.

Rules:
1. Write exactly what is said, in Persian script, keeping the colloquial spoken forms (e.g. «میشه»، «اون»، «یه»، «میخوان»، «اینا») and the speaker's word order. Never convert colloquial speech into formal written Persian. Never summarize, shorten, paraphrase, translate or skip anything — include repetitions, self-corrections, short student questions, answers and interjections.
2. The segment may begin or end in the middle of a sentence; transcribe those partial sentences too.
3. Add natural Persian punctuation (، . ؟ : « ») and start a new paragraph at topic shifts.
4. Speaker labels at every change of speaker, at the start of a line: «استاد:» for the lecturer, «دانشجو:» for a student (use «دانشجو ۱:»، «دانشجو ۲:» when different students are clearly distinguishable).
5. Timestamps: put [mm:ss] — measured from the start of THIS audio segment — at the beginning of every speaker turn and roughly every 30 seconds inside long turns.
6. {terms_rule}
7. Mark unclear words as [نامفهوم], inaudible crosstalk as [همهمه], and non-speech events in brackets, e.g. [خنده]، [سکوت]، [موسیقی].
8. Output only the transcript — no title, preface, notes or commentary."""


class DailyQuota(Exception):
    pass


def api_key():
    for line in open(os.path.join(OMP, ".env"), encoding="utf-8"):
        m = re.match(r"\s*(?:export\s+)?GEMINI_API_KEY\s*=\s*(.*)", line)
        if m:
            return m.group(1).strip().strip("\"'")
    sys.exit("GEMINI_API_KEY not found in ~/.omp/agent/.env")


def ag_creds():
    """Live Antigravity accounts; OMP refreshes these tokens in agent.db, so re-read on every call."""
    db = sqlite3.connect(f"file:{OMP}/agent.db?mode=ro", uri=True)
    try:
        rows = db.execute("select id, data from auth_credentials where provider='google-antigravity' and disabled_cause is null").fetchall()
    finally:
        db.close()
    return {cid: json.loads(data) for cid, data in rows}


def token_refresher(stop):
    """OMP refreshes Antigravity OAuth tokens only when it uses them near expiry; poke it with a one-shot call."""
    cmd = ["omp", "-p", "--no-session", "--no-tools", "--no-skills", "--no-rules", "--no-extensions", "--no-title",
           "--model", "google-antigravity/gemini-3.8-flash", "Reply with the single word OK"]
    while True:
        creds = ag_creds()
        if creds:
            left = min(c.get("expires", 0) for c in creds.values()) / 1000 - time.time()
            if left < 240:
                try:
                    subprocess.run(cmd, capture_output=True, timeout=180)
                except Exception as e:
                    print(f"[refresh] omp call failed: {e!r}", flush=True)
                after = {cid: round((c.get("expires", 0) / 1000 - time.time()) / 60, 1) for cid, c in ag_creds().items()}
                print(f"[refresh] min token life was {left:.0f}s; now minutes left per account: {after}", flush=True)
        if stop.wait(30):
            return


def call(model, parts, gen_cfg, tries=8):
    """Gemini API (API key) generateContent with retry on transient errors."""
    key = api_key()
    body = json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": gen_cfg}).encode()
    last = None
    for k in range(tries):
        req = urllib.request.Request(f"{BASE}/{model}:generateContent", data=body,
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            if e.code == 429 and "PerDay" in err:
                raise DailyQuota(model)
            last = f"HTTP {e.code}: {err[:400]}"
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # network resets, timeouts
            last = repr(e)
        time.sleep(min(120, 6 * 2 ** k))
    raise RuntimeError(last)


def response_text(d):
    cand = d["candidates"][0]
    out = []
    for p in cand.get("content", {}).get("parts", []):
        if p.get("thought"):
            continue
        if "audioTranscription" in p:
            out.append(p["audioTranscription"].get("text", ""))
        elif "text" in p:
            out.append(p["text"])
    return " ".join(out).strip(), cand.get("finishReason")


_rr = itertools.count()


def ag_call(model, parts, gen_cfg, tries=10):
    """Antigravity streamGenerateContent; rotates accounts, skips ones that are rate-limited."""
    cooldown = {}
    last = None
    for k in range(tries):
        creds = ag_creds()
        now = time.time()
        live = [cid for cid, c in creds.items() if c.get("expires", 0) > now * 1000 + 30000 and cooldown.get(cid, 0) <= now]
        if not live:
            last = last or "no live Antigravity token (expired or all cooling down)"
            time.sleep(30)
            continue
        cid = live[next(_rr) % len(live)]
        c = creds[cid]
        body = {"project": c["projectId"], "model": model,
                "request": {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen_cfg},
                "userAgent": "antigravity", "requestId": f"agent-{uuid.uuid4()}"}
        req = urllib.request.Request(f"{AG_BASE}/v1internal:streamGenerateContent?alt=sse", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {c['access']}", "Content-Type": "application/json",
                                              "Accept": "text/event-stream", "User-Agent": AG_UA})
        texts, finish, usage = [], None, {}
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    resp = json.loads(line[5:].strip())
                    resp = resp.get("response", resp)
                    for cand in resp.get("candidates", []):
                        for p in cand.get("content", {}).get("parts", []):
                            if not p.get("thought") and "text" in p:
                                texts.append(p["text"])
                        finish = cand.get("finishReason") or finish
                    usage = resp.get("usageMetadata") or usage
            text = "".join(texts).strip()
            if finish == "STOP" and text:
                return text, finish, usage, cid
            last = f"account {cid}: finish={finish} chars={len(text)}"
        except urllib.error.HTTPError as e:
            last = f"account {cid}: HTTP {e.code}: {e.read().decode('utf-8', 'replace').replace(c['access'], '[redacted]')[:300]}"
            if e.code == 429:
                cooldown[cid] = time.time() + 120
            elif e.code not in (401, 403, 500, 502, 503, 504):
                break
        except Exception as e:
            last = f"account {cid}: {e!r}"
        time.sleep(min(90, 5 * 2 ** k))
    raise RuntimeError(last)


def audio_part(wd, seg):
    with open(os.path.join(wd, seg["file"]), "rb") as f:
        return {"inline_data": {"mime_type": "audio/flac", "data": base64.b64encode(f.read()).decode()}}


def words(t):
    t = re.sub(r"\[[^\]]*\]|\u200c", " ", t)
    t = re.sub(r"^\s*(استاد|دانشجو(?:\s*[۰-۹0-9]+)?)\s*:", " ", t, flags=re.M)
    return re.findall(r"[\w\u0600-\u06FF]+", t)


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["i"]] = r
    return done


def run(kind, wd, model, workers, only):
    segs = json.load(open(os.path.join(wd, "segments.json"), encoding="utf-8"))["segments"]
    if only:
        segs = [s for s in segs if s["i"] in only]
    ctx = json.load(open(os.path.join(wd, "context.json"), encoding="utf-8"))
    prompt = PROMPT_TEMPLATE.format(course=ctx["course"], terms_rule=ctx["terms_rule"])
    out_path = os.path.join(wd, "pass1.jsonl" if kind == "pass1" else f"ag_{model}.jsonl")
    done = load_done(out_path)
    drafts = load_done(os.path.join(wd, "pass1.jsonl"))
    todo = [s for s in segs if s["i"] not in done]
    print(f"{kind} {model}: {len(todo)} to do, {len(done)} done", flush=True)
    lock = threading.Lock()
    quota_out = threading.Event()

    def work(seg):
        if quota_out.is_set():
            return "skipped"
        t0 = time.time()
        if kind == "pass1":
            d = call(model, [audio_part(wd, seg)], {"temperature": 0})
            text, finish = response_text(d)
            usage, acct = d.get("usageMetadata", {}), None
        else:
            text, finish, usage, acct = ag_call(model, [audio_part(wd, seg), {"text": prompt}],
                                                {"temperature": 0.2, "maxOutputTokens": 32768})
        rec = {"i": seg["i"], "start": seg["start"], "end": seg["end"], "model": model, "finish": finish,
               "text": text, "usage": usage, "account": acct, "secs": round(time.time() - t0, 1)}
        extra = ""
        if kind == "ag" and seg["i"] in drafts and drafts[seg["i"]]["text"]:
            a, b = words(drafts[seg["i"]]["text"]), words(text)
            rec["sim_to_pass1"] = round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)
            rec["len_ratio_to_pass1"] = round(len(b) / max(1, len(a)), 3)
            extra = f" sim={rec['sim_to_pass1']} len={rec['len_ratio_to_pass1']}"
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"seg {seg['i']:03d} {finish} {rec['secs']}s words={len(words(text))}{extra}", flush=True)
        return "ok"

    failed, skipped = [], 0
    stop = threading.Event()
    if kind == "ag":
        threading.Thread(target=token_refresher, args=(stop,), daemon=True).start()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, s): s["i"] for s in todo}
        for fu in cf.as_completed(futs):
            try:
                skipped += fu.result() == "skipped"
            except DailyQuota:
                quota_out.set()
                skipped += 1
            except Exception as e:
                failed.append(futs[fu])
                print(f"seg {futs[fu]:03d} FAILED: {e}", flush=True)
    stop.set()
    if quota_out.is_set():
        print(f"daily quota for {model} exhausted; {skipped} segments left for another day", flush=True)
    print("FAILED " + ",".join(map(str, sorted(failed))) if failed else "ALL DONE", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["pass1", "ag"])
ap.add_argument("workdir")
ap.add_argument("--model", default=None)
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--only", default="")
a = ap.parse_args()
model = a.model or ("gemini-3.5-transcribe" if a.kind == "pass1" else "gemini-3.8-flash-high")
run(a.kind, os.path.abspath(a.workdir), model, a.workers, {int(x) for x in a.only.split(",") if x.strip()})
