import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from crafterdojo.agent.deps import crafter_constants as constants

logger = logging.getLogger("deps")


@dataclass(frozen=True)
class ActionRecipe:
    output: str
    action: str  # "mine", "craft", or "place"
    inputs: Dict[str, int] = field(default_factory=dict)
    tools: Tuple[str, ...] = field(default_factory=tuple)      # required inventory tools (e.g., pickaxes)
    stations: Tuple[str, ...] = field(default_factory=tuple)   # required placed stations (e.g., table, furnace)


class CrafterKnowledgeGraph:
    """Builds a deterministic DAG of Crafter item dependencies from the env constants."""

    def __init__(self):
        self.aliases = {"water": "drink"}
        self.recipes = self._build_recipes()
        self._action_priority = {"place": 0, "craft": 1, "mine": 2}
        self.item_caps = {
            name: int((cfg or {}).get("max", 1_000_000_000))
            for name, cfg in (getattr(constants, "items", {}) or {}).items()
        }

    def normalize_item(self, item: str | None) -> str | None:
        if item is None:
            return None
        key = str(item)
        return self.aliases.get(key, key)

    def _cap_quantity(self, item: str, qty: int) -> int:
        try:
            cap = int(self.item_caps.get(item, qty))
        except (TypeError, ValueError):
            cap = qty
        try:
            return min(max(int(qty), 0), cap)
        except (TypeError, ValueError):
            return 0

    def _build_recipes(self) -> dict[str, list[ActionRecipe]]:
        recipes: dict[str, List[ActionRecipe]] = defaultdict(list)

        # Mining/collecting recipes (resource gathering).
        for _, cfg in (constants.collect or {}).items():
            receive = (cfg or {}).get("receive") or {}
            require = (cfg or {}).get("require") or {}
            tools = tuple(require.keys())

            for output in receive:
                recipes[output].append(
                    ActionRecipe(
                        output=output,
                        action="mine",
                        inputs={},
                        tools=tools,
                        stations=(),
                    )
                )

        # Crafting recipes.
        for output, cfg in (constants.make or {}).items():
            uses = self._normalize_counts((cfg or {}).get("uses"))
            stations = tuple((cfg or {}).get("nearby") or ())

            recipes[output].append(
                ActionRecipe(
                    output=output,
                    action="craft",
                    inputs=uses,
                    tools=(),
                    stations=stations,
                )
            )

        # Placement recipes (table/furnace).
        for output, cfg in (constants.place or {}).items():
            uses = self._normalize_counts((cfg or {}).get("uses"))
            recipes[output].append(
                ActionRecipe(
                    output=output,
                    action="place",
                    inputs=uses,
                    tools=(),
                    stations=(),
                )
            )

        return recipes

    def _normalize_counts(self, counts: dict | None) -> dict[str, int]:
        if not counts:
            return {}
        normalized = {}
        for key, value in counts.items():
            try:
                normalized[key] = max(int(value), 0)
            except (TypeError, ValueError):
                continue
        return normalized

    def _select_recipe(self, item: str) -> ActionRecipe | None:
        options = self.recipes.get(item, [])
        if not options:
            return None

        non_place = [r for r in options if r.action != "place"]
        if non_place:
            options = non_place

        def score(recipe: ActionRecipe):
            cost = sum(recipe.inputs.values())
            return (self._action_priority.get(recipe.action, 99), cost, len(recipe.tools) + len(recipe.stations))

        return sorted(options, key=score)[0]

    def plan(
        self,
        target_item: str,
        inventory: dict,
        nearby_table: bool = False,
        nearby_furnace: bool = False,
    ) -> List[Tuple[dict, str]]:
        target_item = self.normalize_item(target_item)
        if target_item is None:
            return []

        inv_counter = Counter()
        for key, value in (inventory or {}).items():
            try:
                inv_counter[str(key)] = max(int(value), 0)
            except (TypeError, ValueError):
                continue

        env_state = {
            "table": bool(nearby_table),
            "furnace": bool(nearby_furnace),
        }

        steps: List[Tuple[dict, str]] = []
        desired_quantity = inv_counter.get(target_item, 0) + 1

        def ensure_station(station: str, stack: tuple[str, ...], *, force: bool = False):
            station = self.normalize_item(station)
            if station is None:
                return
            if env_state.get(station, False) and not force:
                return

            recipe = self._select_recipe(station)
            if recipe is None:
                logger.warning("No placement recipe known for station '%s'", station)
                env_state[station] = True
                return

            ensure_resources(recipe.inputs, stack + (station,))
            consume_inputs(recipe.inputs)
            steps.append(self._format_step(recipe))
            env_state[station] = True

        def ensure_resources(requirements: dict[str, int], stack: tuple[str, ...]):
            while True:
                all_met = True
                for res, needed in requirements.items():
                    needed = self._cap_quantity(res, needed)
                    if inv_counter[res] < needed:
                        ensure_item(res, needed, stack)
                        all_met = False
                        break
                if all_met:
                    return

        def consume_inputs(inputs: dict[str, int]):
            for res, amt in inputs.items():
                inv_counter[res] -= amt

        def ensure_item(item: str, quantity: int, stack: tuple[str, ...]):
            item = self.normalize_item(item)
            if item is None:
                return
            quantity = self._cap_quantity(item, quantity)
            if inv_counter[item] >= quantity:
                return

            if item in stack:
                logger.warning("Cycle detected while planning %s -> %s", " -> ".join(stack), item)
                return

            recipe = self._select_recipe(item)
            if recipe is None:
                logger.warning("No recipe available for item '%s'", item)
                return

            if recipe.action == "place":
                ensure_station(item, stack + (item,))
                return

            if recipe.action == "craft":
                pre_gather = self._collect_tool_less_requirements(
                    item,
                    quantity,
                    stack,
                    inventory=inv_counter,
                    env_state=env_state,
                )
                if pre_gather:
                    ensure_resources(dict(pre_gather), stack + (item,))

                station_inputs: dict[str, int] = {}
                for station in recipe.stations:
                    place_recipe = self._select_recipe(station)
                    if place_recipe and place_recipe.action == "place":
                        for k, v in place_recipe.inputs.items():
                            station_inputs[k] = station_inputs.get(k, 0) + v

                required_resources = dict(recipe.inputs)
                for k, v in station_inputs.items():
                    required_resources[k] = required_resources.get(k, 0) + v

                ensure_resources(required_resources, stack + (item,))

                while inv_counter[item] < quantity:
                    for station in recipe.stations:
                        ensure_station(station, stack + (item,), force=True)

                    ensure_resources(recipe.inputs, stack + (item,))
                    consume_inputs(recipe.inputs)
                    steps.append(self._format_step(recipe))
                    inv_counter[item] += 1
                return

            if recipe.action == "mine":
                for tool in recipe.tools:
                    ensure_item(tool, 1, stack + (item,))

                while inv_counter[item] < quantity:
                    target_amt = self._cap_quantity(item, inv_counter[item] + 1)
                    steps.append(self._format_step(recipe, target_amount=target_amt))
                    inv_counter[item] += 1
                return

            logger.warning("Unsupported action type '%s' for item '%s'", recipe.action, item)

        ensure_item(target_item, desired_quantity, ())
        return steps

    def _format_step(self, recipe: ActionRecipe, *, target_amount: int | None = None) -> Tuple[dict, str]:
        if recipe.action == "mine":
            tools = list(recipe.tools) if recipe.tools else []
            plan = {
                "function": "mine",
                "output_item": recipe.output,
                "tools": tools,
            }
            if target_amount is not None:
                plan["target_count"] = self._cap_quantity(recipe.output, target_amount)
            plan_text = f"mine({repr(recipe.output)}, {repr(tools)})"
            return plan, plan_text

        if recipe.action == "craft":
            inputs = dict(recipe.inputs)
            tools = list(recipe.stations)
            plan = {
                "function": "craft",
                "output_item": recipe.output,
                "input_items": inputs,
                "tools": tools,
            }
            plan_text = f"craft({repr(recipe.output)}, {repr(inputs)}, {repr(tools)})"
            return plan, plan_text

        if recipe.action == "place":
            plan = {"function": "place", "block_item": recipe.output}
            plan_text = f"place({repr(recipe.output)})"
            return plan, plan_text

        plan = {"function": recipe.action, "output_item": recipe.output}
        return plan, f"{recipe.action}({repr(recipe.output)})"

    def _collect_tool_less_requirements(
        self,
        item: str,
        quantity: int = 1,
        stack: tuple[str, ...] = (),
        inventory: dict | Counter | None = None,
        env_state: dict | None = None,
    ) -> Counter:
        item = self.normalize_item(item)
        if item is None:
            return Counter()

        if item in stack:
            return Counter()

        try:
            qty = max(int(quantity), 0)
        except (TypeError, ValueError):
            qty = 0

        inv = Counter()
        for k, v in (inventory or {}).items():
            try:
                inv[str(k)] = max(int(v), 0)
            except (TypeError, ValueError):
                continue

        current_have = inv.get(item, 0)
        if current_have >= qty:
            return Counter()

        env_state = env_state or {}

        if qty == 0:
            return Counter()

        recipe = self._select_recipe(item)
        if recipe is None:
            return Counter()

        stack = stack + (item,)

        if recipe.action == "mine":
            total = Counter()
            missing_qty = max(qty - current_have, 0)
            if not recipe.tools:
                self._update_capped_counter(total, item, missing_qty)
                return total

            for tool in recipe.tools:
                if inv.get(tool, 0) >= 1:
                    continue
                total.update(
                    self._collect_tool_less_requirements(
                        tool,
                        1,
                        stack,
                        inventory=inv,
                        env_state=env_state,
                    )
                )
            return total

        if recipe.action == "craft":
            total = Counter()

            station_inputs: dict[str, int] = {}
            for station in recipe.stations:
                if env_state.get(station, False):
                    continue
                place_recipe = self._select_recipe(station)
                if place_recipe and place_recipe.action == "place":
                    for k, v in place_recipe.inputs.items():
                        station_inputs[k] = station_inputs.get(k, 0) + v

            for res, amt in recipe.inputs.items():
                missing = max(qty * amt - inv.get(res, 0), 0)
                if missing <= 0:
                    continue
                total.update(
                    self._collect_tool_less_requirements(
                        res,
                        missing,
                        stack,
                        inventory=inv,
                        env_state=env_state,
                    )
                )

            for res, amt in station_inputs.items():
                missing = max(qty * amt - inv.get(res, 0), 0)
                if missing <= 0:
                    continue
                total.update(
                    self._collect_tool_less_requirements(
                        res,
                        missing,
                        stack,
                        inventory=inv,
                        env_state=env_state,
                    )
                )

            return total

        if recipe.action == "place":
            total = Counter()
            for res, amt in recipe.inputs.items():
                missing = max(qty * amt - inv.get(res, 0), 0)
                if missing <= 0:
                    continue
                total.update(
                    self._collect_tool_less_requirements(
                        res,
                        missing,
                        stack,
                        inventory=inv,
                        env_state=env_state,
                    )
                )
            return total

        return Counter()

    def _update_capped_counter(self, counter: Counter, item: str, amount: int):
        amount = self._cap_quantity(item, amount)
        if amount <= 0:
            return
        current = counter.get(item, 0)
        cap = self._cap_quantity(item, current + amount)
        counter[item] = cap


class KnowledgeGraphPlanner:
    """Drop-in planner for DEPSAgent that uses the Crafter knowledge graph instead of an LLM."""

    def __init__(self, graph: CrafterKnowledgeGraph | None = None):
        self.graph = graph or CrafterKnowledgeGraph()
        self.tokens_counter = Counter()
        self.reset()

    def reset(self):
        self.plans: List[Tuple[dict, str]] = []
        self.last_inventory: dict = {}
        self.target_item: str | None = None

    def generate_plan(self, info: dict, target_item: str):
        info = info or {}
        self.last_inventory = dict((info.get("inventory") or {}))
        self.target_item = self.graph.normalize_item(target_item)

        self.plans = self.graph.plan(
            target_item=self.target_item,
            inventory=self.last_inventory,
            nearby_table=info.get("nearby_table", False),
            nearby_furnace=info.get("nearby_furnace", False),
        )
        logger.info("Generated KG plan: %s", [p[1] for p in self.plans])

    def report_success(self, info: dict):
        self.last_inventory = dict((info or {}).get("inventory") or {})
        if self.plans:
            completed = self.plans.pop(0)
            logger.info("Plan succeeded: %s", completed[1])

    def report_failure(self, info: dict):
        info = info or {}
        self.last_inventory = dict(info.get("inventory") or {})
        logger.info("Plan failed; replanning from latest state.")
        self.plans = self.graph.plan(
            target_item=self.target_item,
            inventory=self.last_inventory,
            nearby_table=info.get("nearby_table", False),
            nearby_furnace=info.get("nearby_furnace", False),
        )

    @property
    def get_current_goal(self):
        return self.plans[0] if self.plans else (None, None)


class StagedKnowledgeGraphPlanner:
    """Stage-by-stage just-in-time KG planner (hdepv2).

    The vanilla KnowledgeGraphPlanner front-loads ALL sub-goals at episode
    start. For long-horizon tasks like iron_pickaxe / iron_sword / diamond
    that emits ~17–20 sub-goals up front and forces the controller through
    a fixed wood→stone→iron→craft pipeline regardless of what the agent has
    already collected mid-episode.

    Staged planning splits the chain into milestones (e.g., wood_pickaxe →
    stone_pickaxe → iron_pickaxe). At each milestone we plan only that
    milestone from the latest live inventory; when the stage's plans
    exhaust, we advance and regenerate. Failure within a stage replans
    just that stage. Net effect: per-stage minimum-resources planning
    that reacts to the actual inventory after each milestone instead of
    a static 9-wood front-load.
    """

    MILESTONE_CHAIN: dict[str, List[str]] = {
        "wood_pickaxe": ["wood_pickaxe"],
        "wood_sword": ["wood_sword"],
        "stone_pickaxe": ["wood_pickaxe", "stone_pickaxe"],
        "stone_sword": ["wood_pickaxe", "stone_sword"],
        "iron_pickaxe": ["wood_pickaxe", "stone_pickaxe", "iron_pickaxe"],
        "iron_sword": ["wood_pickaxe", "stone_pickaxe", "iron_sword"],
        "diamond": ["wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond"],
    }

    def __init__(self, graph: CrafterKnowledgeGraph | None = None):
        self.graph = graph or CrafterKnowledgeGraph()
        self.tokens_counter = Counter()
        self.reset()

    def reset(self):
        self.plans: List[Tuple[dict, str]] = []
        self.last_inventory: dict = {}
        self.target_item: str | None = None
        self._milestones: List[str] = []
        self._milestone_idx: int = 0

    def _resolve_chain(self, target_item: str) -> List[str]:
        chain = self.MILESTONE_CHAIN.get(target_item)
        if chain is not None:
            return list(chain)
        return [target_item]

    def _current_milestone(self) -> str | None:
        if 0 <= self._milestone_idx < len(self._milestones):
            return self._milestones[self._milestone_idx]
        return None

    def _plan_current_milestone(self, info: dict):
        milestone = self._current_milestone()
        if milestone is None:
            self.plans = []
            return
        info = info or {}
        self.last_inventory = dict(info.get("inventory") or {})
        self.plans = self.graph.plan(
            target_item=milestone,
            inventory=self.last_inventory,
            nearby_table=info.get("nearby_table", False),
            nearby_furnace=info.get("nearby_furnace", False),
        )
        for plan, _ in self.plans:
            plan["stage_target"] = milestone
        logger.info(
            "Generated KG plan for milestone '%s' (idx=%d/%d): %s",
            milestone,
            self._milestone_idx + 1,
            len(self._milestones),
            [p[1] for p in self.plans],
        )

    def _advance_until_plannable(self, info: dict):
        """Advance milestone_idx while current stage planned to no work."""
        info = info or {}
        inventory = info.get("inventory") or {}
        while not self.plans and self._milestone_idx < len(self._milestones) - 1:
            self._milestone_idx += 1
            milestone = self._current_milestone()
            try:
                have = int(inventory.get(milestone, 0) or 0)
            except (TypeError, ValueError):
                have = 0
            if have >= 1 and milestone != self.target_item:
                logger.info(
                    "Skipping milestone '%s' (idx=%d) — already in inventory",
                    milestone,
                    self._milestone_idx + 1,
                )
                continue
            self._plan_current_milestone(info)

    def generate_plan(self, info: dict, target_item: str):
        info = info or {}
        self.target_item = self.graph.normalize_item(target_item)
        self._milestones = self._resolve_chain(self.target_item)
        self._milestone_idx = 0

        # Skip any leading milestones already satisfied by the start inventory.
        inventory = info.get("inventory") or {}
        while self._milestone_idx < len(self._milestones) - 1:
            milestone = self._milestones[self._milestone_idx]
            try:
                have = int(inventory.get(milestone, 0) or 0)
            except (TypeError, ValueError):
                have = 0
            if have >= 1:
                logger.info(
                    "Skipping milestone '%s' at episode start — already in inventory",
                    milestone,
                )
                self._milestone_idx += 1
                continue
            break

        logger.info(
            "Staged KG planner initialized: target='%s' chain=%s starting_idx=%d",
            self.target_item,
            self._milestones,
            self._milestone_idx + 1,
        )
        self._plan_current_milestone(info)
        self._advance_until_plannable(info)

    def report_success(self, info: dict):
        info = info or {}
        self.last_inventory = dict(info.get("inventory") or {})
        if self.plans:
            completed = self.plans.pop(0)
            logger.info("Plan succeeded: %s", completed[1])
        if self._current_milestone_satisfied(info):
            logger.info(
                "Milestone '%s' satisfied; skipping remaining stage plans.",
                self._current_milestone(),
            )
            self.plans = []
        self._advance_until_plannable(info)

    def report_failure(self, info: dict):
        info = info or {}
        self.last_inventory = dict(info.get("inventory") or {})
        if self._current_milestone_satisfied(info):
            logger.info(
                "Milestone '%s' satisfied during failed subgoal; advancing stage.",
                self._current_milestone(),
            )
            self.plans = []
            self._advance_until_plannable(info)
            return
        logger.info(
            "Plan failed; replanning milestone '%s' from latest state.",
            self._current_milestone(),
        )
        self._plan_current_milestone(info)
        self._advance_until_plannable(info)

    @property
    def get_current_goal(self):
        return self.plans[0] if self.plans else (None, None)

    def _current_milestone_satisfied(self, info: dict) -> bool:
        milestone = self._current_milestone()
        if milestone is None:
            return False
        inventory = (info or {}).get("inventory") or {}
        try:
            return int(inventory.get(milestone, 0) or 0) >= 1
        except (TypeError, ValueError):
            return False
