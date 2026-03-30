import gradio as gr
from env.environment import SupportOpsEnv
from inference import smart_policy


def run_simulation():
    env = SupportOpsEnv(seed=42)
    logs = []
    total_scores = []

    for i in range(3):
        obs = env.reset()
        logs.append(f"\n===== TASK {i+1} =====")
        final_score = 0

        for step in range(6):
            action = smart_policy(obs)
            obs, reward, done, _ = env.step(action)

            logs.append(
                f"Step {step+1} | Action: {action.action_type} | Score: {reward.score:.2f}"
            )

            final_score = reward.score

            if done:
                break

        logs.append(f"Final Task Score: {final_score:.2f}")
        total_scores.append(final_score)

    avg = sum(total_scores) / len(total_scores)
    logs.append(f"\n🔥 Average Score: {avg:.2f}")

    return "\n".join(logs)


demo = gr.Interface(
    fn=run_simulation,
    inputs=[],
    outputs="text",
    title="SupportOps OpenEnv Simulator",
    description="Real-world customer support simulation with SLA, tools, and AI agent behavior."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)