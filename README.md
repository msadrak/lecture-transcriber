# lecture-transcriber

Pause-segmented, verbatim Persian lecture transcription pipeline powered by Gemini (via [Oh My Pi](https://github.com/can1357/oh-my-pi) Antigravity OAuth accounts) with optional free-tier `gemini-3.5-transcribe` cross-checking, automated QA heuristics, and Markdown assembly.

## How It Works

1. **Pause-aware segmentation (`segment.py`)**  
   Decodes any `ffmpeg`-readable audio (`.m4a`, `.mp3`, etc.) to `16 kHz` mono PCM, computes `20 ms` RMS energy and a `400 ms` sliding average, and cuts `60–120 s` FLAC chunks at the quietest `400 ms` window so words aren't clipped at boundaries. Also estimates speech seconds per segment and warns if any cut exceeds `25%` of speech level.
2. **Dual-path transcription (`gem_pipeline.py`)**  
   - **Primary (`ag`, default `gemini-3.8-flash-high`):** Rotates across active `google-antigravity` credentials in `~/.omp/agent/agent.db`, refreshes expiring OAuth tokens in the background via `omp`, and transcribes verbatim colloquial Persian with `استاد:` / `دانشجو:` labels, `[mm:ss]` segment-relative timestamps, and course-specific terminology from `context.json`.
   - **Cross-check (`pass1`, `gemini-3.5-transcribe`):** Uses `GEMINI_API_KEY` in `~/.omp/agent/.env` on a sampled subset (e.g. every 5th–6th segment, respecting the 25 req/day free-tier cap) to detect dropped clauses and verify spoken dates/numbers.
   - All runs append incrementally to `.jsonl` files and are fully resumable (`--only` for targeted retries).
3. **Review & QA (`review.py`, `assemble.py --qa-only`)**  
   Checks every segment for `finishReason == "STOP"`, speech-density drops (`< 60%` of median words/speech-minute), 4-gram repetition loops, out-of-order timestamps, and word-count gaps against `pass1`. Includes `scan` for dates/exams/deadlines/`[نامفهوم]` hotspots/Latin terms and `pair` for side-by-side comparison.
4. **Markdown assembly (`assemble.py`)**  
   Normalizes speaker labels, converts segment-relative `[mm:ss]` stamps (including Persian digits) to absolute `[h:mm:ss]` timestamps, drops any backwards timestamp slips (`> 2 s`) without guessing, merges consecutive turns by the same speaker, and writes a formatted Markdown transcript with a Persian legend header.

## Requirements

- **Python 3.12** (`segment.py` uses stdlib `audioop`, removed in Python 3.13; no third-party Python packages required)
- **`ffmpeg`** on `PATH`
- **Oh My Pi (`omp`)** authenticated with `google-antigravity` (`~/.omp/agent/agent.db`)
- *(Optional)* `GEMINI_API_KEY` in `~/.omp/agent/.env` for `pass1` cross-checking

## Usage

```bash
# 1. Segment audio into 60-120 s FLAC chunks
py -3.12 segment.py path/to/lecture.m4a lectures/session01

# 2. Add course context & spelling rules
cp context.example.json lectures/session01/context.json
# Edit lectures/session01/context.json

# 3. Pilot check on 3 segments
py -3.12 gem_pipeline.py ag lectures/session01 --only 0,1,30 --workers 3
py -3.12 review.py lectures/session01 show 0,1,30

# 4. Full run + sampled cross-check
py -3.12 gem_pipeline.py ag lectures/session01 --workers 3
py -3.12 gem_pipeline.py pass1 lectures/session01 --workers 3 --only 0,5,10,15

# 5. QA & review
py -3.12 assemble.py lectures/session01 NUL --qa-only   # use /dev/null on Linux/macOS
py -3.12 review.py lectures/session01 compare
py -3.12 review.py lectures/session01 scan

# 6. Assemble final Markdown transcript
py -3.12 assemble.py lectures/session01 path/to/lecture_transcript.md
```

See [`SKILL.md`](SKILL.md) for the full operator/agent instructions, including boundary checks, date/number tie-breaking, and verification steps.
