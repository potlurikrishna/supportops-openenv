import random
from env.models import Observation, Action, Reward, Message
from env.tasks import TASKS
from env.grader import evaluate_step
from env.utils import refund_api, db_lookup, add_noise

MAX_STEPS = 6


class SupportOpsEnv:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.task = None
        self.steps = 0
        self.done = False
        self.state_data = {}
        self.sla_remaining = 0

    def reset(self):
        self.task = random.choice(TASKS)
        self.steps = 0
        self.done = False
        self.sla_remaining = self.task["sla"]

        noisy_input = add_noise(self.task["input"])

        self.state_data = {
            "category": None,
            "priority": None,
            "resolved": False,
            "escalated": False,
            "tool_result": None,
            "last_tool": None,
            "conversation": [
                Message(role="user", content=noisy_input)
            ]
        }

        return self._get_obs()

    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode finished")

        self.steps += 1
        self.sla_remaining -= 1

        # Apply actions
        if action.action_type == "classify":
            self.state_data["category"] = action.content

        elif action.action_type == "prioritize":
            self.state_data["priority"] = action.content

        elif action.action_type == "respond":
            self.state_data["conversation"].append(
                Message(role="agent", content=action.content or "")
            )

        elif action.action_type == "escalate":
            self.state_data["escalated"] = True

        elif action.action_type == "resolve":
            self.state_data["resolved"] = True
            self.done = True

        elif action.action_type == "refund_api":
            self.state_data["tool_result"] = refund_api(self.task["input"])
            self.state_data["last_tool"] = "refund_api"

        elif action.action_type == "db_lookup":
            self.state_data["tool_result"] = db_lookup(self.task["input"])
            self.state_data["last_tool"] = "db_lookup"

        # Evaluate
        score, breakdown = evaluate_step(self.task, self.state_data, action)

        # ✅ FIXED REWARD LOGIC (CRITICAL)
        if not self.done:
            reward_value = score * 0.6   # stronger intermediate reward
        else:
            reward_value = score         # full reward at end

        # SLA penalty (reduced harshness)
        if self.sla_remaining <= 0:
            reward_value -= 0.1

        # Force episode end at max steps
        if self.steps >= MAX_STEPS:
            self.done = True
            reward_value = score

        # Clamp reward
        reward_value = max(0.0, min(1.0, reward_value))

        return self._get_obs(), Reward(score=reward_value, breakdown=breakdown), self.done, {}

    def state(self):
        return {
            "task": self.task,
            "state": self.state_data,
            "steps": self.steps,
            "sla_remaining": self.sla_remaining
        }

    def _get_obs(self):
        if self.done:
            status = "closed"
        elif self.state_data["escalated"]:
            status = "escalated"
        elif self.steps == 0:
            status = "open"
        else:
            status = "in_progress"

        return Observation(
            ticket_id=self.task["id"],
            conversation=self.state_data["conversation"],
            category=self.state_data["category"],
            priority=self.state_data["priority"],
            status=status,
            steps_remaining=MAX_STEPS - self.steps,
            sla_remaining=self.sla_remaining,
            tool_result=self.state_data["tool_result"]
        )