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

    # 1. CATEGORY (core understanding)
    if state["category"] == expected["category"]:
        score += 0.15
        breakdown["category"] = 1

    # 2. PRIORITY
    if state["priority"] == expected["priority"]:
        score += 0.15
        breakdown["priority"] = 1

    # 3. ESCALATION (strong signal)
    if expected["escalate"]:
        if state["escalated"]:
            score += 0.25
            breakdown["escalation"] = 1
    else:
        if not state["escalated"]:
            score += 0.05
            breakdown["escalation"] = 1

    # 4. TOOL USAGE (VERY IMPORTANT)
    expected_tool = expected.get("tool")

    if expected_tool:
        if state.get("last_tool") == expected_tool:
            score += 0.25
            breakdown["tool"] = 1
    else:
        if state.get("last_tool") is None:
            score += 0.05
            breakdown["tool"] = 1

    # 5. RESOLUTION (final reward, but not dominant)
    if state["resolved"]:
        score += 0.2
        breakdown["resolution"] = 1

    return min(score, 1.0), breakdown