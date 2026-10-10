from lightning.fabric import Fabric

from crafterdojo.agent.vpt.factory import build_vpt_agent


def build(config, env_entry, fabric: Fabric):
    del env_entry
    device = fabric.device

    assert config.weights is not None, "Weights must be provided for evaluation"
    agent = build_vpt_agent(config, device, freeze_value_head=False)
    agent.policy.eval()
    return {"agent": agent}


def main(config, env_entry, episode_id: int, logdir: str, fabric: Fabric, _resources=None):
    if env_entry is None:
        raise ValueError(
            "Evaluation requires an environment config. Pass env=craftax_pixels or env=craftax_dojo."
        )
    env = env_entry(episode_id=episode_id)

    if _resources is None:
        _resources = build(config=config, env_entry=env_entry, fabric=fabric)
    agent = _resources["agent"]
    agent.reset()

    obs, info = env.reset()

    done = False
    while not done:
        action = agent.get_action({"pov": obs["img"]})
        obs, _, done, info = env.step(action)

    env.close()
