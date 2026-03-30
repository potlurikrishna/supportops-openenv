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

## Actions
- classify
- prioritize
- respond
- escalate
- resolve
- refund_api
- db_lookup

## Run

```bash
docker build -t supportops .
docker run supportops

## Observation Space
Contains ticket metadata, conversation history, SLA timers, and tool outputs.

## Action Space
Agents can classify, prioritize, respond, escalate, resolve, and use tools.

## Reward Design
- Partial rewards for correct classification, priority, escalation
- Tool usage reward
- SLA penalty for delays
- Final reward based on correctness

## Baseline Score
~0.80 using rule-based policy