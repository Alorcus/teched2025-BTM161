import random
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..llm import normalize_content

CUSTOMER_SCENARIOS = [
    "You want to order a large latte and a croissant. Be friendly.",
    "You want 2 espressos. You're in a hurry, so keep it brief.",
    "Your last cappuccino was cold and disappointing. You want to complain and get a resolution.",
    "You want to try something new — ask for a recommendation and order based on their suggestion.",
]


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
        else:
            self.scenario = random.choice(CUSTOMER_SCENARIOS)

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
