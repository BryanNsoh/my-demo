# Medical Conversation Analysis Demo

This repository demonstrates a pipeline that uses an LLM to extract and summarize
clinical conversation data. It identifies conditions, medications, instructions, follow-ups,
and resolves ambiguous terms using context. The output is a structured JSON ready for integration.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
