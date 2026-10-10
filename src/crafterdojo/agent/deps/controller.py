import json
import pathlib
from collections import deque

from craftax.craftax_classic.constants import BlockType

from crafterdojo.agent.deps import crafter_constants as constants
from crafterdojo.agent.steve1 import Steve1Agent


DEPS_STEVE1_SKILL_MAP_PATH = str(
    pathlib.Path("data/prompts/deps/deps_steve1_skills.json")
)


class DEPSController:
    def __init__(
        self,
        steve1: Steve1Agent,
    ):
        self.steve1 = steve1
        with open(DEPS_STEVE1_SKILL_MAP_PATH, "rt") as f:
            self.skill_map = json.load(f)

        # Craftax-Classic semantic ids (BlockType.value), used to test whether
        # the cell adjacent to the player can host a placed station/block.
        self._placement_materials = {
            int(BlockType.GRASS.value),
            int(BlockType.PATH.value),
            int(BlockType.SAND.value),
        }

        self.subgoal = None
        self.history = None
        self.current_timestep = 0

        self._skill_queue = deque()
        self._active_skill = None
        self._skill_goal = None
        self._craft_context = None
        self._gather_priority = ("wood", "stone")

    def reset_subgoal(self, subgoal: dict):
        self.subgoal = subgoal
        self.current_timestep = 0
        self.history = None

        self._skill_queue.clear()
        self._active_skill = None
        self._skill_goal = None
        self._craft_context = None

        function = subgoal.get("function")

        if function == "mine":
            output_item = subgoal.get("output_item")
            if output_item:
                self._skill_queue.append(output_item)
            else:
                # legacy fallback for multi-output plans
                output_items = subgoal.get("output_items") or {}
                for item, count in output_items.items():
                    try:
                        repeats = max(int(count), 0)
                    except (TypeError, ValueError):
                        repeats = 0

                    for _ in range(repeats or 1):
                        self._skill_queue.append(item)
        elif function == "place":
            block_item = subgoal.get("block_item")
            if block_item:
                self._skill_queue.append(f"place_{block_item}")
        elif function == "craft":
            self._craft_context = {
                "output_item": subgoal.get("output_item"),
                "input_items": subgoal.get("input_items"),
                "tools": subgoal.get("tools"),
            }

    def get_action(self, obs, info: dict):
        self.current_timestep += 1

        if self.subgoal is None:
            return "noop"

        if self.subgoal["function"] in ("mine", "place"):
            return self._call_steve1(obs, info)

        if self.subgoal["function"] == "craft":
            return self._craft_action(obs, info)

        return "noop"

    def _call_steve1(self, obs, info: dict):
        if self.steve1 is None:
            return "noop"

        if self._active_skill is None:
            while self._skill_queue:
                candidate = self._skill_queue.popleft()
                skill_cfg = self.skill_map.get(candidate, None)
                if skill_cfg is None:
                    continue

                goal_prompt = skill_cfg.get("prompt", candidate)
                self._active_skill = candidate

                self.steve1.reset()
                self.steve1.set_goal(goal_prompt)
                break

        if self._active_skill is None:
            return "noop"

        subgoal_function = (self.subgoal or {}).get("function")
        mask_place = subgoal_function == "mine"
        mask_craft = subgoal_function in ("mine", "place")

        return self.steve1.get_action(
            obs,
            mask_place=mask_place,
            mask_craft=mask_craft,
        )

    def _craft_action(self, obs, info: dict):
        output_item = self.subgoal.get("output_item")
        if not isinstance(output_item, str):
            return self._run_legacy_craft(obs, info)

        if output_item in ("table", "furnace"):
            return self._craft_station(output_item, obs, info)

        requirements = self._compute_craft_requirements(info, output_item)
        required_resources = requirements["required_resources"]

        self._update_active_skill_goal(info, required_resources)

        if self._active_skill is not None:
            return self._call_steve1(obs, info)

        gather_target = self._next_gather_target(required_resources, info)
        if gather_target:
            item, target = gather_target
            if self._start_gather_skill(item, target):
                return self._call_steve1(obs, info)

        station = self._next_station_to_place(requirements["missing_stations"], info)
        if station:
            if self._start_place_skill(station):
                return self._call_steve1(obs, info)

        return f"make_{output_item}"

    def _craft_station(self, station: str, obs, info: dict):
        cost_item, cost_amount = self._station_cost(station)
        if cost_item is None or cost_amount <= 0:
            return "noop"

        required_resources = {cost_item: cost_amount}
        self._update_active_skill_goal(info, required_resources)

        if self._active_skill is not None:
            return self._call_steve1(obs, info)

        if (info or {}).get(f"nearby_{station}", False):
            return "noop"

        if self._get_inventory_count(info, cost_item) < cost_amount:
            if self._start_gather_skill(cost_item, cost_amount):
                return self._call_steve1(obs, info)
            return "noop"

        if self._has_adjacent_space(info) and self._start_place_skill(station):
            return self._call_steve1(obs, info)

        return "noop"

    def _compute_craft_requirements(self, info: dict, output_item: str) -> dict:
        recipe = constants.make.get(output_item, {})
        input_items = self._normalize_counts(
            (self._craft_context or {}).get("input_items")
            or recipe.get("uses", {})
        )

        recipe_tools = recipe.get("nearby", [])
        tools = (self._craft_context or {}).get("tools")
        if tools is None:
            tools = []
        if not isinstance(tools, (list, tuple)):
            tools = [tools]

        full_tools = list(tools)
        for required in recipe_tools or []:
            if isinstance(required, str) and required not in full_tools:
                full_tools.append(required)

        info = info or {}
        required_stations = [t for t in full_tools if isinstance(t, str)]
        missing_stations = [
            station for station in ("table", "furnace")
            if station in required_stations and not info.get(f"nearby_{station}", False)
        ]

        required_resources = dict(input_items)
        if "table" in missing_stations:
            required_resources["wood"] = required_resources.get("wood", 0) + 2
        if "furnace" in missing_stations:
            required_resources["stone"] = required_resources.get("stone", 0) + 1

        return {
            "output_item": output_item,
            "required_resources": required_resources,
            "missing_stations": missing_stations,
        }

    def _next_gather_target(self, required_resources: dict, info: dict):
        for item in self._gather_priority:
            needed = required_resources.get(item, 0)
            if needed <= 0:
                continue
            if self._get_inventory_count(info, item) < needed:
                return item, needed
        return None

    def _next_station_to_place(self, missing_stations, info: dict):
        if not missing_stations:
            return None

        ordered = [s for s in ("table", "furnace") if s in missing_stations]
        for station in ordered:
            cost_item, cost_amount = self._station_cost(station)
            if cost_item is None:
                continue

            if self._get_inventory_count(info, cost_item) < cost_amount:
                continue

            if not self._has_adjacent_space(info):
                continue

            return station

        return None

    def _has_adjacent_space(self, info: dict) -> bool:
        semantic = (info or {}).get("semantic")
        player_pos = (info or {}).get("player_pos")

        if semantic is None or player_pos is None:
            return False

        try:
            px, py = int(player_pos[0]), int(player_pos[1])
        except (TypeError, ValueError, IndexError):
            return False

        height, width = semantic.shape[:2]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = px + dx, py + dy

            if nx < 0 or ny < 0 or nx >= height or ny >= width:
                continue

            cell = int(semantic[nx, ny])
            if cell in self._placement_materials:
                return True

        return False

    def _start_gather_skill(self, item: str, target: int) -> bool:
        if not self._set_active_skill(item):
            return False

        self._skill_goal = {
            "type": "gather",
            "item": item,
            "target": max(int(target or 0), 0),
        }
        return True

    def _start_place_skill(self, station: str) -> bool:
        skill_name = f"place_{station}"
        if not self._set_active_skill(skill_name):
            return False

        self._skill_goal = {
            "type": "place",
            "station": station,
        }
        return True

    def _set_active_skill(self, skill_name: str) -> bool:
        if self.steve1 is None:
            return False

        skill_cfg = self.skill_map.get(skill_name)
        if skill_cfg is None:
            return False

        self._skill_queue.clear()
        self._active_skill = skill_name

        self.steve1.reset()
        self.steve1.set_goal(skill_cfg.get("prompt", skill_name))
        return True

    def _update_active_skill_goal(self, info: dict, required_resources: dict):
        if not self._skill_goal:
            return

        goal_type = self._skill_goal.get("type")
        if goal_type == "gather":
            item = self._skill_goal.get("item")
            target = self._skill_goal.get("target", 0)

            if item:
                if item in required_resources:
                    target = max(int(required_resources.get(item, target) or 0), 0)
                else:
                    target = 0
                self._skill_goal["target"] = target

            if self._get_inventory_count(info, item) >= target:
                self._clear_active_skill_goal()

        elif goal_type == "place":
            station = self._skill_goal.get("station")
            if station and (info or {}).get(f"nearby_{station}", False):
                self._clear_active_skill_goal()

    def _clear_active_skill_goal(self):
        self._active_skill = None
        self._skill_goal = None

    def _station_cost(self, station: str):
        if station == "table":
            return "wood", 2
        if station == "furnace":
            return "stone", 1
        return None, 0

    def _get_inventory_count(self, info: dict, item: str) -> int:
        try:
            return int(((info or {}).get("inventory") or {}).get(item, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _normalize_counts(self, counts) -> dict:
        if not isinstance(counts, dict):
            return {}

        result = {}
        for key, value in counts.items():
            try:
                qty = max(int(value), 0)
            except (TypeError, ValueError):
                continue
            result[key] = qty

        return result

    def _run_legacy_craft(self, obs, info: dict):
        if self.history is None:
            self.history = self._legacy_craft(obs, info)

        try:
            return next(self.history)
        except StopIteration:
            self.history = None
            return "noop"

    def _legacy_craft(self, obs, info: dict):
        output_item = self.subgoal.get("output_item")
        if output_item:
            yield f"make_{output_item}"
            return

        # legacy fallback for multi-output craft actions
        output_items = self.subgoal.get("output_items") or {}
        for k, v in output_items.items():
            try:
                repeats = max(int(v), 0)
            except (TypeError, ValueError):
                repeats = 0

            repeats = repeats or 1
            for _ in range(repeats):
                yield f"make_{k}"

    @property
    def active_skill(self):
        return self._active_skill

    @property
    def has_pending_skill(self) -> bool:
        return (
            self._active_skill is not None
            or bool(self._skill_queue)
            or self.history is not None
            or self._skill_goal is not None
        )
