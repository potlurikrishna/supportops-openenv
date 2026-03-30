TASKS = [
    {
        "id": 1,
        "difficulty": "easy",
        "input": "I was charged twice for my subscription",
        "sla": 4,
        "expected": {
            "category": "billing",
            "priority": "medium",
            "escalate": False,
            "tool": "refund_api"
        }
    },
    {
        "id": 2,
        "difficulty": "medium",
        "input": "App crashes when uploading a file",
        "sla": 3,
        "expected": {
            "category": "technical",
            "priority": "high",
            "escalate": False,
            "tool": "db_lookup"
        }
    },
    {
        "id": 3,
        "difficulty": "hard",
        "input": "Unauthorized transaction from my account",
        "sla": 2,
        "expected": {
            "category": "security",
            "priority": "urgent",
            "escalate": True,
            "tool": None         
        }
    }
]