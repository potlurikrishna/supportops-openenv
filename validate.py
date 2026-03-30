from env.environment import SupportOpsEnv
from env.models import Action


def validate():
    env = SupportOpsEnv(seed=42)

    # Test reset
    obs = env.reset()
    assert obs is not None, "Reset failed"

    # Test step
    for _ in range(3):
        action = Action(action_type="resolve")
        obs, reward, done, _ = env.step(action)

        assert 0.0 <= reward.score <= 1.0, "Reward out of bounds"

    # Test state
    state = env.state()
    assert "task" in state
    assert "state" in state

    print("✅ Validation Passed")


if __name__ == "__main__":
    validate()