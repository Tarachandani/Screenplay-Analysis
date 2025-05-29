## Screenplay and Subtitle Mapping

## Files Overview

2012_script.txt: Original screenplay text file.

2012_script_chunks.txt: Original screenplay divided into equal chunks, generated using screenplay_2_screenplaychunk.py.

2012 (2009) [4K].en-en.srt: Subtitle file in standard SRT format:


> ...  
> Subtitle index.   
> Subtitle start time --> Subtitle end time.   
> Subtitle text.    
> ...
>

## Goal

Use Large Language Models (LLMs) to map corresponding screenplay chunks to movie timestamps. The primary question explored is whether this mapping is one-to-one or many-to-one.

## Usage

Run the script:

> python3 chunks_to_mapping.py

This will attempt to map each subtitle interval to its corresponding screenplay chunk and output the results to subtitle_to_script_chunk_mapping.csv.

Current mapping needs improvement.
