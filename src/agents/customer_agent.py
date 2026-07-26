import json
import re
import random
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..llm import normalize_content


# Single source of truth for customer scenarios.
# Each entry is (short_label, prompt, success_criterion). Order here defines the
# order everywhere (dashboards, notebooks, CLI --scenario index).
# `prompt` tells the customer how to behave; `success_criterion` states what
# counts as actually getting what they came for, and is what `get_feedback`
# rates against — keep the two in sync when adding or editing a scenario.
CUSTOMER_SCENARIO_DEFS: list[tuple[str, str, str]] = [
    (
        "Plain espresso",
        "You want to order a single plain espresso. Not more, not less. Politely decline anything else.",
        "You have your plain espresso.",
    ),
    (
        "Large latte & croissant",
        "You want to order a large latte and a croissant. Be friendly.",
        "You have both the large latte and the croissant.",
    ),
    (
        "2 espressos (hurry)",
        "You want 2 espressos. You're in a hurry, so keep it brief.",
        "You have both espressos, without a long wait or a lot of unnecessary questions.",
    ),
    (
        "Complaint & resolution",
        "Your last cappuccino was cold and disappointing. You want to complain and get a resolution.",
        "Your complaint was actually resolved — a refund, a replacement, or another concrete "
        "remedy. Sympathy alone is not a resolution.",
    ),
    (
        "Ask for recommendation",
        "You want to try something new — ask for a recommendation and order based on their suggestion.",
        "You were recommended a drink, and after you agreed to the recommendation you received "
        "that exact drink. The suggestion on its own is not a result.",
    ),
    (
        "Tea only (stubborn)",
        "You want to order a tea. You do not want anything else and you are stubborn about it — politely but firmly refuse every upsell or alternative.",
        "Either you got your tea, or you were told clearly and politely that tea is not "
        "available. Being talked into something else is not success.",
    ),
    (
        "Buy everything (rich)",
        "You are rich and want to buy everything from the store. Nothing should be left. Keep ordering more items until the staff confirms the store is empty.",
        "You bought everything the store had, and the staff confirmed that nothing is left.",
    ),
]

CUSTOMER_SCENARIO_LABELS: list[str] = [label for label, _, _ in CUSTOMER_SCENARIO_DEFS]
CUSTOMER_SCENARIOS: list[str] = [prompt for _, prompt, _ in CUSTOMER_SCENARIO_DEFS]
CUSTOMER_SCENARIO_SUCCESS: list[str] = [success for _, _, success in CUSTOMER_SCENARIO_DEFS]


def build_default_prompt(scenario_index: int = 0) -> str:
    """Build the full default system prompt for a given scenario index."""
    scenario = CUSTOMER_SCENARIOS[scenario_index] if 0 <= scenario_index < len(CUSTOMER_SCENARIOS) else CUSTOMER_SCENARIOS[0]
    return f"""You are a customer at an AI-powered coffee shop chatting with the staff.

Your goal: {scenario}

Guidelines:
- Keep replies short (1-2 sentences max).
- Be natural, like a real customer texting.
- Respond directly to what the staff last said.
- When your order is confirmed ready OR your complaint is fully resolved, reply with exactly one word: DONE
"""


class CustomerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.history = []
        self.scenario = CUSTOMER_SCENARIOS[0]
        self.custom_prompt: str | None = None
        self.max_turns = 15
        self.turn_count = 0

    def reset(self, scenario_index=None, custom_prompt=None):
        self.history = []
        self.turn_count = 0
        self.custom_prompt = custom_prompt
        if scenario_index is not None and 0 <= scenario_index < len(CUSTOMER_SCENARIOS):
            self.scenario = CUSTOMER_SCENARIOS[scenario_index]
            self.scenario_index = scenario_index
        else:
            self.scenario = random.choice(CUSTOMER_SCENARIOS)
            self.scenario_index = CUSTOMER_SCENARIOS.index(self.scenario)

    def _system_prompt(self):
        if self.custom_prompt:
            return self.custom_prompt
        return f"""You are a customer at an AI-powered coffee shop chatting with the staff.

Your goal: {self.scenario}

Guidelines:
- Keep replies short (1-2 sentences max).
- Be natural, like a real customer texting.
- Respond directly to what the staff last said.
- When your order is confirmed ready OR your complaint is fully resolved, reply with exactly one word: DONE
"""

    def _success_criterion(self) -> str:
        """What counts as getting what the customer came for, for this run.

        `self.scenario` holds the scenario's prompt text (not its index), so we
        look up the parallel success list by matching it. Custom prompts have no
        preset criterion, so fall back to the customer's own opening request.
        """
        default = "you got what you asked for in your opening message"
        if self.custom_prompt:
            return default
        try:
            return CUSTOMER_SCENARIO_SUCCESS[CUSTOMER_SCENARIOS.index(self.scenario)]
        except ValueError:
            return default

    def get_feedback(self) -> dict:
        """Generate a subjective customer satisfaction rating based on the completed conversation."""
        lines = []
        for role, content in self.history:
            if role == "customer":
                lines.append(f"You: {content}")
            elif role == "agent":
                lines.append(f"Staff: {content}")
            elif role == "system_note":
                # Something the customer went through rather than said — e.g.
                # tasting a drink brewed on a dirty machine.
                lines.append(f"[What you experienced: {content}]")
        history_text = "\n".join(lines)
        messages = [
            SystemMessage(content=(
                "You are evaluating the coffee shop interaction from the customer's perspective.\n\n"
                f"Customer goal:\n{self.custom_prompt or self.scenario}\n\n"
                f"Expected successful outcome:\n{self._success_criterion()}\n\n"
                "Evaluate the interaction using a score from 0.0 to 1.0.\n\n"
                "Judge only by what the transcript shows. Do not assume an order was placed, drinks "
                "were brewed, or anything was handed over unless the staff confirmed it happened.\n\n"
                "Scoring rubric:\n"
                "- 0.9-1.0: The customer received exactly what they wanted, and the interaction was smooth.\n"
                "- 0.7-0.8: The customer received an acceptable alternative or the issue was resolved with only minor friction.\n"
                "- 0.4-0.6: The customer partially achieved their goal, but there were noticeable problems "
                "such as confusion, repeated corrections, delays, or unclear communication.\n"
                "- 0.1-0.3: The customer did not receive the requested item or accepted substitute, the "
                "order failed, or the agent took an inappropriate action.\n"
                "- 0.0: The interaction completely failed, contradicted the customer intent, or became nonsensical.\n\n"
                "Important constraints:\n"
                "- Do not give a high score if the customer did not receive the requested item or an explicitly accepted alternative.\n"
                "- If the agent recommended or processed an item that does not exist on the menu, penalize the score.\n"
                "- If the customer had to repeat or correct the request multiple times, penalize the score.\n"
                "- If the final outcome is unclear, penalize the score.\n"
                "- Normal clarifying or confirming questions (asking about size, extras, or payment) are good "
                "service, not friction — do not lower the score for them. If the goal was fully achieved, do "
                "not go below 0.7 unless there were real problems: wrong items, menu hallucinations, repeated "
                "corrections, or delays the customer actually complained about.\n\n"
                'Return only valid JSON in this exact format:\n'
                '{"score": 0.85, "reason": "short explanation"}'
            )),
            HumanMessage(content=f"Your conversation:\n{history_text}\n\nYour rating:"),
        ]
        response = self.llm.invoke(messages)
        raw = normalize_content(response.content).strip()

        score = None
        reason = "No reason provided."
        try:
            text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start : end + 1])
                score = float(parsed.get("score", -1))
                reason = parsed.get("reason", "No reason provided.")
        except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
            pass

        if score is not None and 0.0 <= score <= 1.0:
            return {
                "feedback_score": round(score, 2),
                "feedback_reason": reason,
                "raw_feedback_response": raw,
                "valid": True,
            }

        return {
            "feedback_score": 0.5,
            "feedback_reason": "Fallback used because the model response was invalid.",
            "raw_feedback_response": raw,
            "valid": False,
        }

    def inject_experience(self, text: str):
        """Inject a mid-conversation experience note (e.g. contaminated coffee)."""
        self.history.append(("system_note", text))

    def get_initial_message(self):
        """Generate the opening message to kick off the conversation."""
        self.turn_count = 0
        messages = [
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content="Write your opening message to the coffee shop staff to start the conversation."),
        ]
        response = self.llm.invoke(messages)
        text = normalize_content(response.content).strip()
        self.history.append(("customer", text))
        return text

    def respond_to(self, agent_message):
        """Return the customer's next message, or None to end the conversation."""
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            return None

        self.history.append(("agent", agent_message))

        messages = [SystemMessage(content=self._system_prompt())]
        for role, content in self.history:
            if role == "customer":
                messages.append(AIMessage(content=content))
            elif role == "system_note":
                messages.append(SystemMessage(content=f"[Experience: {content}]"))
            else:
                messages.append(HumanMessage(content=content))

        response = self.llm.invoke(messages)
        text = normalize_content(response.content).strip()

        if text.upper() == "DONE" or (len(text) <= 10 and "DONE" in text.upper()):
            return None

        self.history.append(("customer", text))
        return text
