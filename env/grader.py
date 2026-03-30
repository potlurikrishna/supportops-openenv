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

    # CATEGORY (IMPORTANT)
    if state["category"] == expected["category"]:
        score += 0.2
        breakdown["category"] = 1

    # PRIORITY
    if state["priority"] == expected["priority"]:
        score += 0.15
        breakdown["priority"] = 1

    # ESCALATION (HIGH IMPACT)
    if expected["escalate"]:
        if state["escalated"]:
            score += 0.25
            breakdown["escalation"] = 1
    else:
        if not state["escalated"]:
            score += 0.1
            breakdown["escalation"] = 1

    # TOOL USAGE (CRITICAL)
    expected_tool = expected.get("tool")

    if expected_tool:
        if state.get("last_tool") == expected_tool:
            score += 0.25
            breakdown["tool"] = 1
    else:
        if state.get("last_tool") is None:
            score += 0.1
            breakdown["tool"] = 1

    # RESOLUTION (FINAL BOOST)
    if state["resolved"]:
        score += 0.2
        breakdown["resolution"] = 1

    return min(score, 1.0), breakdown