import random


def refund_api(ticket: str):
    if "charged" in ticket.lower() or "payment" in ticket.lower():
        return {"status": "success", "refund_id": "R123"}
    return {"status": "denied"}


def db_lookup(ticket: str):
    return {
        "user_id": 123,
        "history": "premium_user",
        "previous_issues": 2
    }


def add_noise(text: str):
    variants = [
        text,
        text.replace("you", "u"),
        text.replace("please", "plz"),
        text + " 😡",
        "Hola " + text,
        "नमस्ते " + text
    ]
    return random.choice(variants)