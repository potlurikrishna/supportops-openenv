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

    # --------------------
    # CATEGORY (core)
    # --------------------
    if state["category"] == expected["category"]:
        score += 0.3
        breakdown["category"] = 1

    # --------------------
    # PRIORITY
    # --------------------
    if state["priority"] == expected["priority"]:
        score += 0.2
        breakdown["priority"] = 1

    # --------------------
    # ESCALATION (FIXED)
    # --------------------
    if expected["escalate"]:
        if state["escalated"]:
            score += 0.3
            breakdown["escalation"] = 1
        else:
            score -= 0.2   # ❗ penalty added
    else:
        if not state["escalated"]:
            score += 0.1
            breakdown["escalation"] = 1

    # --------------------
    # TOOL USAGE
    # --------------------
    expected_tool = expected.get("tool")

    if expected_tool:
        if state.get("last_tool") and state["last_tool"] == expected_tool:
            score += 0.2
            breakdown["tool"] = 1
    else:
        # No tool expected → reward if agent avoids unnecessary tools
        if state.get("last_tool") is None:
            score += 0.1
            breakdown["tool"] = 1

    # --------------------
    # RESOLUTION (STRICTER)
    # --------------------
    if state["resolved"]:
        # only reward if agent performed well
        if score >= 0.8:   # stricter threshold
            score += 0.2
            breakdown["resolution"] = 1

    return max(0.0, min(score, 1.0)), breakdown