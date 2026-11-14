"""Cut a lecture recording into 60-120 s segments at the quietest moments (Python 3.12 stdlib + ffmpeg).

Usage: py -3.12 segment.py <audio> <workdir>
Writes <workdir>/seg/seg_NNN.flac and <workdir>/segments.json:
  {"audio": <abs path>, "duration": s,
   "segments": [{"i", "start", "end", "file" (relative to workdir), "speech" (s of detected speech)}]}
"""
import concurrent.futures as cf, json, os, subprocess, sys, warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # noqa: E402  (removed in Python 3.13; run with 3.12)

SR = 16000
FRAME = 320  # 20 ms
WIN = 20  # 400 ms quiet window
MIN_LEN, MAX_LEN = 60.0, 120.0


def main(audio, wd):
    audio = os.path.abspath(audio)
    os.makedirs(os.path.join(wd, "seg"), exist_ok=True)
    pcm = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", audio, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    rms = [audioop.rms(pcm[i:i + FRAME * 2], 2) for i in range(0, len(pcm), FRAME * 2)]
    total = len(pcm) / 2 / SR
    del pcm

    pref = [0]
    for v in rms:
        pref.append(pref[-1] + v)
    smooth = [(pref[i + WIN] - pref[i]) / WIN for i in range(len(rms) - WIN)]

    def quietest(t0, t1):
        a, b = int(t0 * 50), min(len(smooth), int(t1 * 50))
        i = min(range(a, b), key=smooth.__getitem__)
        return (i + WIN / 2) / 50, smooth[i]

    cuts, start = [], 0.0
    while total - start > MAX_LEN * 1.25:
        cut, level = quietest(start + MIN_LEN, start + MAX_LEN)
        cuts.append((start, cut, level))
        start = cut
    cuts.append((start, total, None))

    ordered = sorted(rms)
    noise, speech_level = ordered[len(ordered) // 10], ordered[len(ordered) * 7 // 10]
    threshold = noise + 0.2 * (speech_level - noise)

    def speech_seconds(s, e):
        return sum(1 for v in rms[int(s * 50):int(e * 50)] if v > threshold) / 50

    def cut_file(k):
        s, e, _ = cuts[k]
        rel = f"seg/seg_{k:03d}.flac"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                        "-i", audio, "-ac", "1", "-ar", str(SR), "-c:a", "flac", os.path.join(wd, rel)], check=True)
        return rel

    with cf.ThreadPoolExecutor(8) as ex:
        files = list(ex.map(cut_file, range(len(cuts))))
    segments = [{"i": k, "start": round(s, 2), "end": round(e, 2), "file": f, "speech": round(speech_seconds(s, e), 1)}
                for k, ((s, e, _), f) in enumerate(zip(cuts, files))]
    with open(os.path.join(wd, "segments.json"), "w", encoding="utf-8") as f:
        json.dump({"audio": audio, "duration": round(total, 2), "segments": segments}, f, ensure_ascii=False, indent=0)

    levels = [lv for _, _, lv in cuts[:-1]]
    loud = [round(s) for s, e, lv in cuts[:-1] if lv > 0.25 * speech_level]
    lens = [e - s for s, e, _ in cuts]
    print(f"{len(cuts)} segments over {total / 60:.1f} min; length {min(lens):.0f}-{max(lens):.0f} s; "
          f"speech level {speech_level}, loudest cut {max(levels, default=0):.0f}; cuts above 25% of speech level at: {loud or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
