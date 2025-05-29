import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# --- CONFIG ---
SCRIPT_CHUNKS_PATH = "2012_script_chunks.txt"
SRT_PATH = '2012 (2009) [4K].en-en.srt'
VIDEO_DURATION_SECONDS = 9649  # <-- Set this to your movie's duration in seconds
CHUNK_WINDOW = 2  # How many chunks before/after to include in the prompt
OUTPUT_CSV = "subtitle_to_script_chunk_mapping.csv"

# --- LOAD MODEL ---
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

# --- PARSE SCRIPT CHUNKS ---
with open(SCRIPT_CHUNKS_PATH, "r", encoding="utf-8") as f:
    script = f.read()
chunks = re.split(r"###script_chunk \d+##", script)[1:]  # skip preamble
num_chunks = len(chunks)

# --- PARSE SRT ---
def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        srt = f.read()
    entries = re.findall(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.+?)(?=\n\n|\Z)",
        srt, re.DOTALL
    )
    return [
        {
            "index": int(idx),
            "start": start,
            "end": end,
            "text": text.replace('\n', ' ').strip()
        }
        for idx, start, end, text in entries
    ]

subs = parse_srt(SRT_PATH)

# --- TIME UTILS ---
def srt_time_to_seconds(t):
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

# --- MAIN LOOP ---
results = []
for sub in subs:
    # Estimate which chunk this subtitle is closest to
    sub_time = (srt_time_to_seconds(sub["start"]) + srt_time_to_seconds(sub["end"])) / 2
    approx_chunk_idx = int((sub_time / VIDEO_DURATION_SECONDS) * num_chunks)
    approx_chunk_idx = max(0, min(num_chunks-1, approx_chunk_idx))

    # Select window of chunks
    start_idx = max(0, approx_chunk_idx - CHUNK_WINDOW)
    end_idx = min(num_chunks, approx_chunk_idx + CHUNK_WINDOW + 1)
    window_chunks = chunks[start_idx:end_idx]

    chunk_list = "\n".join([f"{i+start_idx+1}. {c.strip()[:300]}" for i, c in enumerate(window_chunks)])

    prompt = f"""<s>[INST] You are given a movie script split into numbered chunks and a subtitle with timestamp.
Return the number of the script chunk that best matches the subtitle.

Subtitle:
Time: {sub['start']} --> {sub['end']}
Text: {sub['text']}

Script Chunks:
{chunk_list}

Which chunk number best matches the subtitle? Return only the number. [/INST]
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        temperature=0.7,
        repetition_penalty=1.1
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the number from the response
    match = re.search(r"\b(\d+)\b", response.split("[/INST]")[-1])
    chunk_num = int(match.group(1)) if match else None

    results.append({
        "subtitle_index": sub["index"],
        "start": sub["start"],
        "end": sub["end"],
        "subtitle_text": sub["text"],
        "matched_chunk_index": chunk_num,
        "matched_chunk_text": chunks[chunk_num-1].strip() if chunk_num and 1 <= chunk_num <= num_chunks else ""
    })

# --- SAVE RESULTS ---
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")