import torch
import numpy as np
import os
import pickle
from lightning.fabric import Fabric

from crafterdojo.agent.steve1.agent import Steve1Agent
from crafterdojo.model.crafterclip import load_mineclip_wconfig


class GTPlanningAgentForCondition:
    def __init__(self, steve1: Steve1Agent):
        self.steve1 = steve1 

        self.cur_goal = None
        self.prev_action = None


    def _get_next_goal(self, info):
        inventory = info["inventory"]
        nearby_table = info["nearby_table"]

        if inventory["wood_pickaxe"] == 0:
            if inventory["wood"] < 3 and not nearby_table:
                return "obtain tree"
            return "craft wood pickaxe"
        
        if inventory["stone_pickaxe"] == 0:
            if inventory["stone"] < 1:
                return "obtain stone"
            if inventory["wood"] < 3 and not nearby_table:
                return "obtain tree"
            return "craft stone pickaxe"

        if inventory["wood"] < 3 and not nearby_table:
            return "obtain tree"

        return "craft wood sword"

    def get_action(self, obs, info):
        next_goal = self._get_next_goal(info)
        if next_goal != self.cur_goal:
            self.cur_goal = next_goal
            self.steve1.reset()
            self.steve1.set_goal(next_goal)

        action = self.steve1.get_action(obs)
        self.prev_action = action
        return action


class GTPlanningAgentForT1:
    def __init__(self, steve1: Steve1Agent):
        self.steve1 = steve1
        self.place_sapling = False
        self.cur_goal = None

        self.place_timesteps = 0

    def get_action(self, obs, info):
        next_goal = self._get_next_goal(info)
        if next_goal != self.cur_goal:
            self.cur_goal = next_goal
            self.steve1.reset()
            self.steve1.set_goal(next_goal)

        action = self.steve1.get_action(obs)
        if action == 10:
            self.place_sapling = True

        if self.place_sapling:
            self.place_timesteps += 1

        return action

    def _get_next_goal(self, info):
        inventory = info["inventory"]

        if self.place_timesteps > 301:
            return "obtain plant"

        if self.place_sapling:
            return "stay"

        if inventory["sapling"] > 0:
            return "place plant"
 
        return "obtain sapling"


class GTPlannerAgentForT2:
    def __init__(self, steve1: Steve1Agent):
        self.steve1 = steve1
        self.cur_goal = None

        self.phase = 0

    def _get_next_goal(self, info):
        inventory = info["inventory"]

        if self.phase == 0:  # for place plant
            if inventory["sapling"] <= 0:
                return "obtain sapling"
            
            return "place plant"

        if self.phase == 1: # for place table
            if inventory["wood"] < 2:
                return "obtain wood"
            
            return "place table"
        
        assert False, "phase error"

    def get_action(self, obs, info):
        next_goal = self._get_next_goal(info)
        if next_goal != self.cur_goal:
            self.cur_goal = next_goal
            self.steve1.reset()
            self.steve1.set_goal(next_goal)

        action = self.steve1.get_action(obs)

        if self.phase == 0 and action == 10:
            self.phase = 1

        return action


class GTPlannerAgentForT3:
    def __init__(self, steve1: Steve1Agent):
        self.steve1 = steve1
        self.cur_goal = None

    def _get_next_goal(self, info):
        inventory = info["inventory"]
        nearby_table = info["nearby_table"]

        if inventory["wood_sword"] == 0:
            if inventory["wood"] < 3 and not nearby_table:
                return "obtain tree"
            return "craft wood sword"

        return "obtain sapling"


    def get_action(self, obs, info):
        next_goal = self._get_next_goal(info)
        if next_goal != self.cur_goal:
            self.cur_goal = next_goal
            self.steve1.reset()
            self.steve1.set_goal(next_goal)

        action = self.steve1.get_action(obs)

        return action


class GTPlannerAgentForT4:
    def __init__(self, steve1: Steve1Agent):
        self.steve1 = steve1
        self.cur_goal = None

        self.phase = 0
        self.wood_pickaxe_at_phase0 = 0
        self.stones_at_phase1 = 0

    def _get_next_goal(self, info):
        inventory = info["inventory"]
        nearby_table = info["nearby_table"]

        if self.phase == 0:  # for obtain coal
            # First, we need to craft wood pickaxe
            if inventory["wood_pickaxe"] == 0:
                if inventory["wood"] < 3 and not nearby_table:
                    return "obtain tree"
                return "craft wood pickaxe"

            if inventory["coal"] == 0:
                return "obtain coal"

            self.phase = 1
            self.wood_pickaxe_at_phase0 = inventory["wood_pickaxe"]

        if self.phase == 1:  # for wood pickaxe again
            if inventory["wood_pickaxe"] <= self.wood_pickaxe_at_phase0:
                if inventory["wood"] < 3 and not nearby_table:
                    return "obtain tree"
                return "craft wood pickaxe"

            self.phase = 2
            self.stones_at_phase1 = inventory["stone"]

        if self.phase == 2:  # for obtain stone
            if inventory["stone"] <= self.stones_at_phase1:
                return "obtain stone"

            return "place stone"

        assert False, "phase error"

    def get_action(self, obs, info):
        next_goal = self._get_next_goal(info)
        if next_goal != self.cur_goal:
            self.cur_goal = next_goal
            self.steve1.reset()
            self.steve1.set_goal(next_goal)

        action = self.steve1.get_action(obs)

        return action


@torch.no_grad()
def eval_entry(config, env_entry, episode_id: int, logdir: str, fabric: Fabric):
    device = fabric.device

    mineclip = load_mineclip_wconfig(device)
    mineclip.eval()
    
    steve1_agent = Steve1Agent(
        config.steve1.model,
        config.steve1.weights,
        config.steve1.prior_weights,
        config.steve1.cond_scale,
        mineclip,
        device,
    )

    agent = GTPlanningAgentForT1(steve1_agent)

    env = env_entry(episode_id=episode_id)

    done = False
    obs, info = env.reset()

    plans = []
    while not done:
        action = agent.get_action(obs, info)
        plans.append(agent.cur_goal)

        obs, _, done, info = env.step(action)

    plan_root = os.path.join(logdir, "gt_plan")
    os.makedirs(plan_root, exist_ok=True)

    with open(os.path.join(plan_root, f"gt_plan_{episode_id:04d}.pkl"), "wb") as f:
        pickle.dump(plans, f)

    env.close()
