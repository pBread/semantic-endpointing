# Semantic Endpointing (WIP)

This repo explores end-of-turn detection for voice AI. Traditional systems usually decide a turn is over by waiting for a fixed amount of silence or by using simple voice activity thresholds. That works, but it either interrupts people when they pause to think or it waits too long after a clear, complete statement.

Semantic endpointing uses a model to analyze the meaning and context of the transcript and estimate whether the speaker is finished. The output is a likelihood that the user is done speaking. That likelihood is then used to drive a dynamic response timer: respond quickly when language looks complete, wait longer when language suggests continuation, and choose an intermediate delay when it is unclear.

This mirrors how people time their replies, using what was said and the brief pause that follows it, not just the length of the pause alone.

## Getting Started

### Setup

### Run Application

#### Start Ngrok Tunnel

For convenience, there's a script that will start Ngrok from the `HOSTNAME` in the env variables.

```bash
chmod +x scripts/ngrok.bash
./scripts/ngrok.bash
```

#### Start Server

```bash
uv run server/main.py
```

## What this demo does and does not do

### Limitations

This demo is simply designed to show a semantic endpointing flow, not a full endpointing implementation.

- Silence is determined with a simple amplitude check. There are much better ways of doing silence detection but they are beyond the scope of this project.
