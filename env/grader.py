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

    # Category (core understanding)
    if state["category"] == expected["category"]:
        score += 0.3
        breakdown["category"] = 1

    # Priority (important but less than category)
    if state["priority"] == expected["priority"]:
        score += 0.2
        breakdown["priority"] = 1

    # Escalation (critical for hard task)
    if expected["escalate"]:
        if state["escalated"]:
            score += 0.3
            breakdown["escalation"] = 1
    else:
        if not state["escalated"]:
            score += 0.1
            breakdown["escalation"] = 1

    expected_tool = expected.get("tool")

    if expected_tool:
        if state.get("last_tool") == expected_tool:
            score += 0.2
            breakdown["tool"] = 1
    else:
        # No tool expected → reward if agent avoids unnecessary tools
        if state.get("last_tool") is None:
            score += 0.1
            breakdown["tool"] = 1

    # Resolution (only if everything else is correct)
    if state["resolved"]:
        if score >= 0.7:  # only reward resolution if agent did well
            score += 0.2
            breakdown["resolution"] = 1

    return min(score, 1.0), breakdown