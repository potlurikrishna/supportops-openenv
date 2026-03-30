def evaluate_step(task, state, action):
    expected = task["expected"]

    score = 0.0
    breakdown = {
        "category": 0,
        "priority": 0,
        "escalation": 0,
        "tool": 0,
        "resolution": 0
    }

    # Category (progress reward)
    if state["category"] == expected["category"]:
        score += 0.1
        breakdown["category"] = 1

    # Priority
    if state["priority"] == expected["priority"]:
        score += 0.1
        breakdown["priority"] = 1

    # Escalation (strong reward)
    if expected["escalate"] and state["escalated"]:
        score += 0.2
        breakdown["escalation"] = 1

    # Tool usage (important signal)
    if state.get("last_tool") == expected.get("tool"):
        score += 0.2
        breakdown["tool"] = 1

    # Resolution (reward partial + final success)
    if state["resolved"]:
        score += 0.2
        breakdown["resolution"] = 1

    return min(score, 1.0), breakdown