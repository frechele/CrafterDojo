import enum
import json
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
from craftax.craftax_classic.constants import BlockType

from crafterdojo.agent.deps import crafter_constants as constants
from crafterdojo.agent.deps.controller import DEPSController
from crafterdojo.agent.deps.knowledge_graph import KnowledgeGraphPlanner

DEPS_STEVE1_SKILL_MAP_PATH = str(
    pathlib.Path("data/prompts/deps/deps_steve1_skills.json")
)
logger = logging.getLogger("deps")


@dataclass
class DEPSObservation:
    pixel_obs: any
    target_item: str


class SurvivalManager:
    def __init__(
        self,
        controller: "DEPSController",
        skill_timeouts: dict,
        threshold: int = 3,
        safe_level: int = 4,
        settle_steps: int = 2,
        health_threshold: int | None = None,
        health_safe_level: int | None = None,
        threat_radius: int = 4,
        flee_steps: int = 8,
    ):
        self.controller = controller
        self.skill_timeouts = skill_timeouts or {}
        self.skill_map = getattr(controller, "skill_map", {}) or {}
        self.threshold = max(int(threshold), 0)
        self.safe_level = max(int(safe_level), self.threshold + 1)
        self.settle_steps = max(int(settle_steps), 1)
        self.health_threshold = max(
            int(health_threshold if health_threshold is not None else threshold), 0
        )
        self.health_safe_level = max(
            int(health_safe_level if health_safe_level is not None else safe_level),
            self.health_threshold + 1,
        )
        self.threat_radius = max(int(threat_radius), 0)
        self.flee_steps = max(int(flee_steps), 1)
        self._max_values = {
            name: int((cfg or {}).get("max", 0) or 0)
            for name, cfg in (getattr(constants, "items", {}) or {}).items()
        }
        self._walkable_blocks = {
            int(BlockType.GRASS.value),
            int(BlockType.PATH.value),
            int(BlockType.SAND.value),
        }

        self.goal = None
        self.steps = 0
        self._settle_counter = 0

    def reset(self):
        self.goal = None
        self.steps = 0
        self._settle_counter = 0
        if self.controller and self.controller.steve1:
            self.controller.steve1.reset()

    def maybe_step(self, obs, info: dict):
        inventory = (info or {}).get("inventory")
        if inventory is None:
            return None, False, None
        inventory = inventory or {}

        if self.goal is None:
            detected = self._detect_need(info, inventory)
            if detected is None:
                return None, False, None
            self._start_goal(detected)

        status, current_value = self._goal_status(inventory)
        stat = self.goal["stat"]

        if status == "success":
            logger.info(
                "Survival recovered %s: %s -> %s",
                stat,
                self.goal["start"],
                current_value,
            )
            self.reset()
            return None, True, True

        timeout = self.goal.get("timeout", 0)
        if timeout and self.steps >= timeout:
            logger.info(
                "Survival attempt timed out for %s after %s steps",
                stat,
                self.steps,
            )
            self.reset()
            return None, True, False

        action = self._heuristic_action(info)
        if action is not None:
            self.steps += 1
            return action, False, None

        action = self.controller.steve1.get_action(
            obs,
            mask_place=True,
            mask_craft=True,
        )
        self.steps += 1
        return action, False, None

    def _detect_need(self, info: dict, inventory: dict):
        health = self._get_value(inventory, "health")
        nearest_threat = self._nearest_threat_distance(info)
        if health <= self.health_threshold or (
            nearest_threat is not None
            and nearest_threat <= self.threat_radius
            and health < self.health_safe_level
        ):
            return {
                "stat": "health",
                "skill": "health",
                "start": health,
                "mode": "recover_health",
            }

        priorities = [
            ("drink", "drink"),
            ("food", "cow"),
            ("energy", "sleep"),
        ]

        candidates = []
        for idx, (stat, skill) in enumerate(priorities):
            value = self._get_value(inventory, stat)
            if value <= self.threshold:
                candidates.append((value, idx, stat, skill))

        if not candidates:
            return None

        value, _, stat, skill = min(candidates, key=lambda x: (x[0], x[1]))
        return {"stat": stat, "skill": skill, "start": value}

    def _start_goal(self, goal: dict):
        stat = goal["stat"]
        skill = goal["skill"]
        start_value = max(int(goal.get("start", 0) or 0), 0)

        max_value = self._max_values.get(stat, 0) or 0
        if stat == "health":
            target = self.health_safe_level
            timeout = self.flee_steps + 30
            prompt = "go sleep"
        else:
            target = max(start_value + 1, self.safe_level)
            timeout = int(self.skill_timeouts.get(skill, 0) or 0)
            prompt = (self.skill_map.get(skill) or {}).get("prompt", skill)

        if stat == "energy" and max_value:
            target = max_value
        elif stat != "health" and max_value:
            target = min(target, max_value)

        self.goal = {
            "stat": stat,
            "skill": skill,
            "start": start_value,
            "target": target,
            "timeout": timeout,
            "mode": goal.get("mode"),
        }
        self.steps = 0
        self._settle_counter = 0

        logger.info(
            "Starting survival plan for %s (value=%s -> target=%s) via %s",
            stat,
            start_value,
            target,
            skill,
        )

        if self.controller and self.controller.steve1:
            self.controller.steve1.reset()
            self.controller.steve1.set_goal(prompt)

    def _goal_status(self, inventory: dict):
        stat = self.goal["stat"]
        current = self._get_value(inventory, stat)

        if stat == "energy":
            if current > self.goal["start"] and current >= self.goal["target"]:
                self._settle_counter += 1
            else:
                self._settle_counter = 0

            if self._settle_counter >= self.settle_steps:
                return "success", current
            return "active", current

        if stat == "health":
            if current >= self.goal["target"]:
                return "success", current
            if current <= 0:
                return "failed", current
            return "active", current

        if current >= self.goal["target"] and current > self.goal["start"]:
            return "success", current

        return "active", current

    def _heuristic_action(self, info: dict):
        if not self.goal or self.goal.get("mode") != "recover_health":
            return None

        threat_action = self._flee_action(info)
        if threat_action is not None:
            logger.info("Survival health flee action: %s", threat_action)
            return threat_action

        inventory = (info or {}).get("inventory") or {}
        if self._get_value(inventory, "drink") <= self.threshold:
            self._switch_skill("drink")
            return None
        if self._get_value(inventory, "food") <= self.threshold:
            self._switch_skill("cow")
            return None
        if self._get_value(inventory, "energy") < self._max_values.get("energy", 9):
            return "sleep"

        return "noop"

    def _switch_skill(self, skill: str):
        if self.goal:
            self.goal["skill"] = skill
            self.goal["timeout"] = int(self.skill_timeouts.get(skill, 0) or 0)
        skill_cfg = self.skill_map.get(skill) or {}
        prompt = skill_cfg.get("prompt", skill)
        if self.controller and self.controller.steve1:
            self.controller.steve1.reset()
            self.controller.steve1.set_goal(prompt)

    def _flee_action(self, info: dict):
        player_pos = (info or {}).get("player_pos")
        if player_pos is None:
            return None

        try:
            pos = np.asarray(player_pos, dtype=np.int32)
        except (TypeError, ValueError):
            return None

        threats = self._hostile_positions(info)
        if not threats:
            return None

        nearest = min(self._manhattan(pos, threat) for threat in threats)
        if nearest > self.threat_radius:
            return None

        semantic = (info or {}).get("semantic")
        moves = {
            "move_left": np.array([0, -1], dtype=np.int32),
            "move_right": np.array([0, 1], dtype=np.int32),
            "move_up": np.array([-1, 0], dtype=np.int32),
            "move_down": np.array([1, 0], dtype=np.int32),
        }

        best_action = None
        best_score = None
        for action, delta in moves.items():
            candidate = pos + delta
            if not self._is_walkable(candidate, semantic):
                continue

            min_dist = min(self._manhattan(candidate, threat) for threat in threats)
            score = (min_dist, self._manhattan(candidate, pos))
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _hostile_positions(self, info: dict):
        threats = []
        for prefix in ("zombies", "skeletons", "arrows"):
            positions = (info or {}).get(f"{prefix}_position")
            masks = (info or {}).get(f"{prefix}_mask")
            if positions is None or masks is None:
                continue

            positions = np.asarray(positions)
            masks = np.asarray(masks, dtype=bool)
            for position, mask in zip(positions, masks):
                if mask:
                    threats.append(np.asarray(position, dtype=np.int32))
        return threats

    def _nearest_threat_distance(self, info: dict) -> int | None:
        player_pos = (info or {}).get("player_pos")
        if player_pos is None:
            return None
        threats = self._hostile_positions(info)
        if not threats:
            return None
        pos = np.asarray(player_pos, dtype=np.int32)
        return min(self._manhattan(pos, threat) for threat in threats)

    def _is_walkable(self, pos, semantic) -> bool:
        if semantic is None:
            return True
        semantic = np.asarray(semantic)
        x, y = int(pos[0]), int(pos[1])
        if x < 0 or y < 0 or x >= semantic.shape[0] or y >= semantic.shape[1]:
            return False
        return int(semantic[x, y]) in self._walkable_blocks

    def _manhattan(self, a, b) -> int:
        return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))

    def _get_value(self, inventory: dict, key: str) -> int:
        try:
            return max(int(inventory.get(key, 0) or 0), 0)
        except (TypeError, ValueError):
            return 0


class DEPSStatus(enum.Enum):
    EXECUTING = enum.auto()
    FAILED = enum.auto()
    COMPLETED = enum.auto()


class DEPSAgent:
    def __init__(
        self,
        planner: KnowledgeGraphPlanner,
        controller: DEPSController,
        survival: dict | None = None,
    ):
        self.planner = planner
        self.controller = controller

        self.current_task = None
        self.current_subgoal = None
        self.current_plan_text = None
        self._active_skill = None
        self._skill_step = 0
        self._skill_timeout = 0
        self._skill_timeouts = self._load_skill_timeouts()
        self._plan_step_count = 0
        self._plan_start_inventory = {}
        self.survival = survival or {}
        self.survival_manager = None

        enable_survival = bool((self.survival or {}).get("enable", False))
        if enable_survival:
            self.survival_manager = SurvivalManager(
                controller=self.controller,
                skill_timeouts=self._skill_timeouts,
                threshold=self.survival.get("threshold", 3),
                safe_level=self.survival.get("safe_level", 4),
                settle_steps=self.survival.get("settle_steps", 2),
                health_threshold=self.survival.get("health_threshold"),
                health_safe_level=self.survival.get("health_safe_level"),
                threat_radius=self.survival.get("threat_radius", 4),
                flee_steps=self.survival.get("flee_steps", 8),
            )

    def reset(self, obs: DEPSObservation, info: dict):
        self.planner.reset()
        if self.survival_manager:
            self.survival_manager.reset()
        controller_reset = getattr(self.controller, "reset_episode", None)
        if callable(controller_reset):
            controller_reset()
        self._set_goal_hook(info, obs.target_item)

    def get_action(self, obs: DEPSObservation, info: dict):
        if self.current_task != obs.target_item:
            logger.info("New task: %s", obs.target_item)
            self._set_goal_hook(info, obs.target_item)

        survival_action = self._maybe_handle_survival(obs, info)
        if survival_action is not None:
            action = survival_action
            if not isinstance(action, str):
                action = constants.actions[action.item()]
            logger.info("Survival action taken: %s", action)
            return action

        self.update_plan(info)

        if self.current_subgoal is None:
            return "noop"

        self._plan_step_count += 1
        action = self.controller.get_action(obs.pixel_obs, info)
        self._sync_skill_state()

        if self._should_timeout(info):
            return "noop"

        if not isinstance(action, str):
            action = constants.actions[action.item()]
        logger.info("Action taken: %s", action)
        return action

    def update_plan(self, info: dict):
        if self.current_subgoal is None:
            return

        if self._subgoal_succeeded(info):
            self._advance_plan(success=True, info=info)
        elif self._subgoal_failed(info):
            self._advance_plan(success=False, info=info)

    def _set_goal_hook(self, info: dict, target_item: str):
        self.current_task = target_item
        self.current_timestep = 0

        self.planner.generate_plan(info, target_item)
        self._assign_current_subgoal(info)

    def _advance_plan(self, success: bool, info: dict, reason: str | None = None):
        self._log_plan_completion("success" if success else "failure", info, reason=reason)

        if success:
            self.planner.report_success(info)
        else:
            self.planner.report_failure(info)

        self._assign_current_subgoal(info)

    def _subgoal_succeeded(self, info: dict) -> bool:
        if self.current_subgoal is None:
            return False

        if info and info.get("is_success") is True:
            return True

        inventory = (info or {}).get("inventory")
        if inventory is None:
            return False

        base_inventory = self.planner.last_inventory or {}
        function = self.current_subgoal.get("function")
        stage_target = self.current_subgoal.get("stage_target")
        if stage_target:
            try:
                if int(inventory.get(stage_target, 0) or 0) >= 1:
                    return True
            except (TypeError, ValueError):
                pass

        if function in ("mine", "craft"):
            output_item = self.current_subgoal.get("output_item")
            if output_item:
                target_count = self.current_subgoal.get("target_count")
                if target_count is not None:
                    try:
                        current_count = int(inventory.get(output_item, 0) or 0)
                    except (TypeError, ValueError):
                        current_count = 0
                    return current_count >= target_count

                if function == "craft" and output_item in ("table", "furnace"):
                    return info.get(f"nearby_{output_item}", False)

                try:
                    base_count = max(int(base_inventory.get(output_item, 0) or 0), 0)
                except (TypeError, ValueError):
                    base_count = 0

                try:
                    current_count = int(inventory.get(output_item, 0) or 0)
                except (TypeError, ValueError):
                    current_count = 0

                return current_count >= base_count + 1

            # legacy fallback for multi-output plans
            outputs = self.current_subgoal.get("output_items") or {}
            if not outputs:
                return False

            for item, count in outputs.items():
                if count is None:
                    continue
                required = max(int(count), 0)
                if inventory.get(item, 0) < base_inventory.get(item, 0) + required:
                    return False
            return True

        if function == "place":
            block_item = self.current_subgoal.get("block_item")
            if not block_item:
                return False

            return info.get(f"nearby_{block_item}", False)

        return False

    def _subgoal_failed(self, info: dict) -> bool:
        if info and info.get("is_success") is False:
            return True

        if self.current_subgoal is None:
            return False

        inventory = (info or {}).get("inventory")
        if inventory is None:
            return False

        if self._plan_step_count == 0:
            return False

        if not self.controller.has_pending_skill and not self._subgoal_succeeded(info):
            return True

        return False

    def _reset_skill_tracking(self):
        self._active_skill = None
        self._skill_step = 0
        self._skill_timeout = 0

    def _sync_skill_state(self):
        controller_skill = self.controller.active_skill
        if controller_skill != self._active_skill:
            self._active_skill = controller_skill
            self._skill_step = 0
            self._skill_timeout = self._get_skill_timeout(controller_skill)

    def _should_timeout(self, info: dict) -> bool:
        if self._active_skill is None:
            return False

        self._skill_step += 1

        if self._skill_timeout and self._skill_step > self._skill_timeout:
            self._advance_plan(success=False, info=info, reason="timeout")
            return True

        return False

    def _load_skill_timeouts(self) -> dict:
        try:
            with open(DEPS_STEVE1_SKILL_MAP_PATH, "rt") as f:
                skill_map = json.load(f)
        except FileNotFoundError:
            logger.warning("Skill timeout map not found; defaulting to zero")
            return {}

        return {
            name: int((cfg or {}).get("timeout", 0) or 0)
            for name, cfg in skill_map.items()
        }

    def _get_skill_timeout(self, skill: str | None) -> int:
        if skill is None:
            return 0

        base_timeout = self._skill_timeouts.get(skill, 0)
        subgoal = self.current_subgoal or {}

        output_item = subgoal.get("output_item")
        if isinstance(output_item, str):
            return self._skill_timeouts.get(output_item, base_timeout)

        # legacy fallback
        output_items = subgoal.get("output_items") or {}
        for item in output_items:
            return self._skill_timeouts.get(item, base_timeout)

        return base_timeout

    def _assign_current_subgoal(self, info: dict):
        while True:
            plan, plan_text = self.planner.get_current_goal
            self.current_subgoal = plan
            self.current_plan_text = plan_text
            self._reset_skill_tracking()

            if plan is None:
                self._plan_step_count = 0
                self._plan_start_inventory = {}
                logger.info("No remaining plans")
                return

            self._plan_start_inventory = self._snapshot_inventory(info)
            self._plan_step_count = 0

            # Skip plans that are already satisfied by the current inventory/state.
            if info and self._subgoal_succeeded(info):
                self._log_plan_completion("success", info, reason="already satisfied")
                self.planner.report_success(info)
                continue

            self.controller.reset_subgoal(plan)
            self._log_plan_start()
            return

    def _log_plan_start(self):
        logger.info("Starting plan: %s", self.current_plan_text or self.current_subgoal)
        logger.info("Inventory before: %s", self._plan_start_inventory)

    def _log_plan_completion(self, outcome: str, info: dict, reason: str | None = None):
        end_inventory = self._snapshot_inventory(info)

        if outcome == "success":
            headline = "Plan succeeded"
        else:
            detail = f" ({reason})" if reason else ""
            headline = f"Plan failed{detail}"

        logger.info("%s: %s", headline, self.current_plan_text or self.current_subgoal)
        logger.info("Inventory after: %s", end_inventory)
        logger.info("Plan duration: %s steps", self._plan_step_count)

    def _snapshot_inventory(self, info: dict) -> dict:
        inventory = (info or {}).get("inventory") or {}
        return dict(inventory)

    def _maybe_handle_survival(self, obs: DEPSObservation, info: dict):
        if not self.survival_manager:
            return None

        action, finished, _ = self.survival_manager.maybe_step(obs.pixel_obs, info)
        if finished:
            self._reset_skill_tracking()
            self._plan_step_count = 0
            self._plan_start_inventory = self._snapshot_inventory(info)
            if self.current_subgoal is not None:
                self.controller.reset_subgoal(self.current_subgoal)
        return action
