---
title: supportops-openenv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.10.0
python_version: '3.10'
app_file: app.py
pinned: false
---

# SupportOps OpenEnv

## Overview
A real-world customer support simulation environment for training AI agents.

## Features
- Multi-turn conversation memory
- SLA deadlines with penalties
- Tool usage (refund API, DB lookup)
- Noisy multilingual inputs

## Tasks
1. Billing issue (easy)
2. Technical issue (medium)
3. Security breach (hard)

## Run
```bash
docker build -t supportops .
docker run supportops