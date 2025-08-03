# Semantic Endpointing (WIP)

A demonstration of semantic endpointing for voice AI agents using machine learning to detect when users have completed their speaking turn.

## What is Semantic Endpointing?

Traditional endpointing uses fixed silence detection, i.e. wait X milliseconds of silence, then assume the user is done. Semantic endpointing analyzes the meaning of what the user said to determine if they're finished speaking.

Examples:

- "Hi, how can I help you?" → Complete (clear question)
- "The name of that is, umm..." → Incomplete (user is thinking)
- "Thank you, that's exactly what I needed." → Complete (statement of satisfaction)
- "Well, I was looking into..." → Incomplete (user continuing thought)

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

- Silence is determined with a simple amplitude check.
