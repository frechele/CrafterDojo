import ast
import logging
import pathlib
from collections import Counter
from enum import StrEnum

from crafterdojo.agent.deps.planner_utils import (
    inventory_description,
    nearby_description,
    resolve_template,
)


PROMPTS_ROOT = pathlib.Path("data/prompts/deps")
DEPS_SYSTEM_PROMPT_PATH = str(PROMPTS_ROOT / "system_prompt.txt")
DEPS_PLAN_PROMPT_PATH = str(PROMPTS_ROOT / "plan_prompt.txt")
DEPS_EXPLAIN_PROMPT_PATH = str(PROMPTS_ROOT / "explain_prompt.txt")
CRAFTER_KNOWLEDGE_PATH = "data/prompts/crafter_knowledge.txt"
CRAFTER_ITEM_LIST_PATH = str(PROMPTS_ROOT / "supported_items.txt")
STEVE1_SKILL_MAP_PATH = str(PROMPTS_ROOT / "deps_steve1_skills.json")
logger = logging.getLogger("deps")


class PlannerFailure(Exception):
    """Raised when the LLM planner cannot produce a usable plan."""

    def __init__(self, reason: str, raw_response: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


class DEPSAction(StrEnum):
    mine = "output_item, tools"
    craft = "output_item, input_items, tools"
    place = "block_item"

    @staticmethod
    def get_action_description(prefix: str | None = None) -> str:
        prefix = prefix or ""
        results = []
        for action in DEPSAction:
            line = f"{prefix}- {action.name}({action.value})"
            results.append(line)
        return '\n'.join(results)


def response_print(response, title: str):
    separator = f"{'='*20} [{title}] {'='*20}"
    logger.info("%s\n%s\n%s", separator, response, separator)


class DEPSPlanner:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        embed_crafter: bool = False,
        **kwargs,
    ):
        # Imported lazily so the KG planner doesn't pull in the LLM stack.
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            **kwargs,
        )

        with open(CRAFTER_ITEM_LIST_PATH, "rt") as f:
            supported_items = [item.strip() for item in f.readlines()]
            supported_items_str = ", ".join(supported_items)

        with open(DEPS_SYSTEM_PROMPT_PATH, "rt") as f:
            system_prompt = f.read().strip()
            action_desc = DEPSAction.get_action_description(prefix=' ' * 4)
            self.system_prompt = resolve_template(
                system_prompt,
                SUPPORTED_ITEMS=supported_items_str,
                ACTION_DESC=action_desc,
            )

        if embed_crafter:
            with open(CRAFTER_KNOWLEDGE_PATH, "rt") as f:
                crafter_knowledge = f.read().strip()
            self.system_prompt += "\n\n" + crafter_knowledge

        response_print(self.system_prompt, "System Prompt")

        with open(DEPS_PLAN_PROMPT_PATH, "rt") as f:
            self.plan_prompt_template = f.read().strip()

        with open(DEPS_EXPLAIN_PROMPT_PATH, "rt") as f:
            self.explain_prompt_template = f.read().strip()

    def reset(self):
        from langchain_core.messages import SystemMessage

        self.messages = [SystemMessage(content=self.system_prompt)]

        self.plans = []
        self.last_inventory = {}

        self.tokens_counter = Counter()

        self.target_item = None
        logger.info("DEPSPlanner reset")

    def generate_plan(self, info: dict, target_item: str) -> str:
        self.last_inventory = info.get("inventory", {}).copy()

        inventory_desc = inventory_description(info)
        nearby_desc = nearby_description(info)

        plan_prompt = resolve_template(
            self.plan_prompt_template,
            TARGET_ITEM=target_item,
            INVENTORY_DESC=inventory_desc,
            NEARBY_DESC=nearby_desc,
        )
        response_print(plan_prompt, "Plan Prompt")

        generated_plan = self._invoke_llm(plan_prompt)
        self.plans = self._parse_plan(generated_plan)

        self.target_item = target_item
        logger.info("Generated plan list: %s", [p[1] for p in self.plans])

        if not self.plans:
            raise PlannerFailure(
                "empty_plan_after_initial_generation", raw_response=generated_plan
            )

    def report_success(self, info: dict):
        assert self.plans, "No plan available to report success."
        self.last_inventory = info.get("inventory", {}).copy()

        _, plan_str = self.plans.pop(0)
        logger.info("Plan succeeded: %s", plan_str)

    def report_failure(self, info: dict):
        self.last_inventory = info.get("inventory", {}).copy()

        _, plan_str = self.plans[0]
        logger.info("Plan failed; explaining: %s", plan_str)
        explain_prompt = self._generate_explain_prompt(info)
        self._invoke_llm(explain_prompt)

        replan_prompt = "Based on the explanation, please provide a revised plan to achieve the original goal.\n"
        replan_prompt += resolve_template(
            self.plan_prompt_template,
            TARGET_ITEM=self.target_item,
            INVENTORY_DESC=inventory_description(info),
            NEARBY_DESC=nearby_description(info),
        )
        response_print(replan_prompt, "Replan Prompt")

        generated_plan = self._invoke_llm(replan_prompt)
        self.plans = self._parse_plan(generated_plan)
        logger.info("Plan updated after failure: %s", [p[1] for p in self.plans])

        if not self.plans:
            raise PlannerFailure(
                "empty_plan_after_failure_replan", raw_response=generated_plan
            )

    def _maybe_replan_after_success(self, completed_plan: str, info: dict):
        inventory_desc = inventory_description(info)
        nearby_desc = nearby_description(info)
        remaining_plans = "\n".join([p[1] for p in self.plans]) if self.plans else "(no remaining steps)"

        prompt = (
            "A plan step just succeeded:\n"
            f"{completed_plan}\n\n"
            f"Current inventory: {inventory_desc}.{nearby_desc}\n"
            "Remaining planned steps:\n"
            f"{remaining_plans}\n\n"
            "If the remaining plan is still valid, reply exactly with \"ok\".\n"
            "If you need to adjust the plan, reply with \"replan:\" with reason to replan and then provide a full updated plan in the same Python code format to obtain "
            f"{self.target_item}. Remember each mine/craft action yields exactly one output_item."
        )

        response = self._invoke_llm(prompt).strip()
        decision = response.splitlines()[0].strip().lower() if response else ""

        if decision == "ok":
            return

        if not decision.startswith("replan"):
            logger.info(
                "Ignoring success-check response without explicit replan prefix: %s",
                response[:500],
            )
            return

        plan_text = "\n".join(response.splitlines()[1:]).strip()

        if not plan_text:
            return

        updated_plans = self._parse_plan(plan_text)
        if updated_plans:
            self.plans = updated_plans
            logger.info("Plan updated after success: %s", [p[1] for p in self.plans])

    @property
    def get_current_goal(self) -> dict:
        return self.plans[0] if self.plans else (None, None)

    def _parse_plan(self, plan_str: str):
        try:
            return self._parse_plan_inner(plan_str)
        except Exception as e:
            logger.warning(
                "Plan parse crashed (%s); treating as empty plan. Raw: %s",
                e,
                plan_str,
            )
            return []

    def _parse_plan_inner(self, plan_str: str):
        plans = []

        # merge multi-line calls into single lines before parsing
        merged_lines = []
        buffer = ""
        paren_depth = 0
        for raw_line in plan_str.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            starts_with_action = any(
                line.startswith(action.name + '(') for action in DEPSAction
            )

            if not buffer and not starts_with_action:
                continue

            if starts_with_action and not buffer:
                buffer = line
            else:
                buffer += " " + line

            paren_depth += line.count('(') - line.count(')')

            if paren_depth <= 0 and buffer:
                merged_lines.append(buffer)
                buffer = ""
                paren_depth = 0

        # parse merged lines
        for line in merged_lines:
            line = line.replace("null", "None")  # gpt-oss output sometimes uses JSON null

            if '#' in line:
                line = line[:line.index('#')].strip()

            function_str = line.split('(')[0]
            try:
                call_expr = ast.parse(line, mode="eval").body
            except SyntaxError:
                continue

            if not isinstance(call_expr, ast.Call):
                continue

            try:
                positional_args = [ast.literal_eval(arg) for arg in call_expr.args]
                keyword_args = {
                    kw.arg: ast.literal_eval(kw.value)
                    for kw in call_expr.keywords
                    if kw.arg is not None
                }
            except (ValueError, SyntaxError) as e:
                logger.warning("literal_eval failed on line %r: %s", line, e)
                continue

            try:
                parameters = [param.strip() for param in DEPSAction[function_str].value.split(',')]
            except KeyError:
                continue
            plan = {"function": function_str}

            for param in parameters:
                if positional_args:
                    plan[param] = positional_args.pop(0)
                elif param in keyword_args:
                    plan[param] = keyword_args[param]
                elif param == "output_item" and "output_items" in keyword_args:
                    plan[param] = keyword_args["output_items"]
                else:
                    plan[param] = None

            normalized_steps = self._normalize_plan_entries(plan, line)
            plans.extend(normalized_steps)

        return plans

    def _normalize_plan_entries(self, plan: dict, line: str):
        function = plan.get("function")

        if function in ("mine", "craft"):
            output_item = plan.get("output_item")

            if isinstance(output_item, dict):
                return self._expand_output_dict(function, output_item, plan, line)

            normalized_plan = dict(plan)
            normalized_plan["output_item"] = output_item
            normalized_line = self._format_action_line(function, normalized_plan, line)
            return [(normalized_plan, normalized_line)]

        return [(plan, line)]

    def _expand_output_dict(self, function: str, output_dict: dict, plan: dict, line: str):
        steps = []

        for item, count in output_dict.items():
            try:
                qty = max(int(count), 0)
            except (TypeError, ValueError):
                qty = 0

            qty = qty or 1

            for _ in range(qty):
                expanded_plan = dict(plan)
                expanded_plan["output_item"] = item
                normalized_line = self._format_action_line(function, expanded_plan, line)
                steps.append((expanded_plan, normalized_line))

        return steps

    def _format_action_line(self, function: str, plan: dict, fallback_line: str) -> str:
        output_item = plan.get("output_item")
        tools = plan.get("tools")
        input_items = plan.get("input_items")

        if function == "mine":
            return f"mine({repr(output_item)}, {repr(tools)})"

        if function == "craft":
            return f"craft({repr(output_item)}, {repr(input_items)}, {repr(tools)})"

        if function == "place":
            return f"place({repr(plan.get('block_item'))})"

        return fallback_line

    def _invoke_llm(self, prompt: str) -> str:
        from langchain_core.messages import AIMessage, HumanMessage

        self.messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(self.messages)
        response_print(response.content, "LLM Response")
        self.messages.append(AIMessage(content=response.content))

        # collect usage stats
        usage = getattr(response, "usage_metadata", None) or {}
        for k, v in usage.items():
            if isinstance(v, int):
                self.tokens_counter[k] += v
        logger.info("Usage stats: %s", dict(self.tokens_counter))

        return response.content

    def _generate_explain_prompt(self, info: dict) -> str:
        inventory_desc = inventory_description(info)
        nearby_desc = nearby_description(info)

        _, plan_str = self.plans[0]

        explain_prompt = resolve_template(
            self.explain_prompt_template,
            PLAN_STR=plan_str,
            INVENTORY_DESC=inventory_desc,
            NEARBY_DESC=nearby_desc,
        )
        response_print(explain_prompt, "Explain Prompt")
        return explain_prompt
