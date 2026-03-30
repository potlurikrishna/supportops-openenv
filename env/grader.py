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

    # CATEGORY
    if state["category"] == expected["category"]:
        score += 0.2
        breakdown["category"] = 1

    # PRIORITY
    if state["priority"] == expected["priority"]:
        score += 0.2
        breakdown["priority"] = 1

    # ESCALATION
    if expected["escalate"]:
        if state["escalated"]:
            score += 0.2
            breakdown["escalation"] = 1
    else:
        if not state["escalated"]:
            score += 0.1
            breakdown["escalation"] = 1

    # TOOL (IMPORTANT)
    if expected.get("tool") == state.get("last_tool"):
        score += 0.3
        breakdown["tool"] = 1

    # RESOLUTION (SMALL FINAL BOOST)
    if state["resolved"]:
        score += 0.1
        breakdown["resolution"] = 1

    return min(score, 1.0), breakdown