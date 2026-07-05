"""
Temporal guardrail evaluator using DECLARE constraints from Chiariello et al. (2025).

Supports all core DECLARE constraint patterns:
- RespondedExistence, Coexistence 
- Choice, ExclusiveChoice
- Response, Precedence
- AlternateResponse, AlternatePrecedence
- ChainResponse, ChainPrecedence

All constraints are binary (relating exactly two activities).
Multi-activity chains are represented by combining multiple binary constraints.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from .temporal_constraints import TemporalConstraint, ConstraintType
from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.temporal_guardrail")

# Global trace storage - keyed by thread_id (shared across agents)
_traces: Dict[str, 'ExecutionTrace'] = {}


class ExecutionTrace:
    """Tracks the execution history for DECLARE constraint checking."""
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: List[Dict[str, Any]] = []
        
        # Track pending responses: antecedent -> list of pending expectations
        self.pending_responses: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Track completed responses for audit
        self.completed_responses: List[Dict[str, Any]] = []
        
        # Session state
        self.session_active = True
        self.session_end_time: Optional[datetime] = None

        self.choice_constraints: List['TemporalConstraint'] = []

        # Administrative tools that should be recorded but not enforced
        self.admin_tools = {
            'transfer_to_agent', 'get_order'
        }
    
    def add_event(self, tool_name: str, tool_call_id: str, timestamp: datetime, context: Dict):
        """Record a tool execution event."""
        if tool_name in self.admin_tools:
            return
        
        self.events.append({
            'tool_name': tool_name,
            'tool_call_id': tool_call_id,
            'timestamp': timestamp,
            'context': context,
            'order': len(self.events) + 1
        })
        
            
        logger.debug(f"Trace {self.thread_id}: recorded {tool_name} as event {len(self.events)}")
    
    
    def add_pending_response(self, antecedent: str, consequent: str, timestamp: datetime, tool_call_id: str, constraint_type: str):
        """Record that we're expecting a response to occur."""
        # Check if this exact constraint is already pending
        for pending_list in self.pending_responses.values():
            for pending in pending_list:
                if pending['antecedent'] == antecedent and pending['consequent'] == consequent:
                    logger.debug(f"Trace {self.thread_id}: {antecedent}->{consequent} already pending, skipping duplicate")
                    return
        
        self.pending_responses[antecedent].append({
            'antecedent': antecedent,
            'consequent': consequent,
            'timestamp': timestamp,
            'tool_call_id': tool_call_id,
            'order': len(self.events),
            'constraint_type': constraint_type
        })
        logger.debug(f"Trace {self.thread_id}: expecting {consequent} after {antecedent} ({constraint_type})")


    def fulfill_response(self, consequent: str, timestamp: datetime, tool_call_id: str) -> Optional[Dict]:
        """Mark a response as fulfilled. Returns the fulfilled pending response if found."""
        for antecedent, pending_list in self.pending_responses.items():
            for i, pending in enumerate(pending_list):
                if pending['consequent'] == consequent:
                    fulfilled = pending.copy()
                    fulfilled['fulfilled_at'] = timestamp
                    fulfilled['fulfilled_by_call_id'] = tool_call_id
                    fulfilled['response_time'] = (timestamp - pending['timestamp']).total_seconds()
                    
                    self.completed_responses.append(fulfilled)
                    self.pending_responses[antecedent].pop(i)
                    if not self.pending_responses[antecedent]:
                        del self.pending_responses[antecedent]
                    
                    logger.debug(f"Trace {self.thread_id}: fulfilled {consequent} after {antecedent} in {fulfilled['response_time']:.2f}s")
                    return fulfilled
        
        return None
    
    def get_unfulfilled_responses(self) -> Dict[str, List[Dict]]:
        """Get all pending responses that were never fulfilled."""
        return dict(self.pending_responses)
    
    def has_unfulfilled_responses(self) -> bool:
        """Check if there are any pending responses."""
        return len(self.pending_responses) > 0
    
    def end_session(self):
        """Called when conversation/agent session ends."""
        self.session_active = False
        self.session_end_time = datetime.now()
        
        # Log unfulfilled responses
        if self.pending_responses:  
            for antecedent, pending_list in self.pending_responses.items():
                for pending in pending_list:
                    constraint_type = pending.get('constraint_type', 'UNKNOWN')
                    logger.info(
                        f"UNFULFILLED {constraint_type}: {antecedent} → {pending['consequent']} "
                    )
        else:
            logger.debug(f"Trace {self.thread_id}: session ended with 0 unfulfilled responses - all constraints satisfied")

        for constraint in self.choice_constraints:
            a_occurred = self.has_occurred(constraint.antecedent)
            b_occurred = self.has_occurred(constraint.consequent)
            
            if constraint.constraint_type == ConstraintType.EXCLUSIVE_CHOICE:
                if not a_occurred and not b_occurred:
                    logger.info(
                        f"UNFULFILLED EXCLUSIVE CHOICE: Neither {constraint.antecedent} NOR {constraint.consequent} were called"
                    )
            elif constraint.constraint_type == ConstraintType.CHOICE:
                if not a_occurred and not b_occurred:
                    logger.info(
                        f"UNFULFILLED CHOICE: Neither {constraint.antecedent} NOR {constraint.consequent} were called"
                    )

    def get_occurrence_order(self, tool_name: str) -> List[int]:
        """Get the occurrence order numbers for a tool."""
        return [e['order'] for e in self.events if e['tool_name'] == tool_name]
    
    def has_occurred(self, tool_name: str) -> bool:
        """Check if a tool has occurred."""
        return any(e['tool_name'] == tool_name for e in self.events)
    
    def get_occurrence_count(self, tool_name: str) -> int:
        """Get how many times a tool has occurred."""
        return sum(1 for e in self.events if e['tool_name'] == tool_name)
    
    def get_last_occurrence(self, tool_name: str) -> Optional[Dict]:
        """Get the last occurrence of a tool."""
        for e in reversed(self.events):
            if e['tool_name'] == tool_name:
                return e
        return None
    
    def get_events_between(self, start_tool: str, end_tool: str) -> List[Dict]:
        """Get all events that occurred between start_tool and end_tool."""
        start_order = self.get_occurrence_order(start_tool)
        end_order = self.get_occurrence_order(end_tool)
        
        if not start_order or not end_order:
            return []
        
        start_idx = start_order[0] - 1
        end_idx = end_order[-1] - 1
        
        return self.events[start_idx + 1:end_idx]
    
    def get_last_n_events(self, n: int) -> List[Dict]:
        """Get the last N events."""
        return self.events[-n:] if n <= len(self.events) else self.events


def get_or_create_trace(thread_id: str, guardrail: 'TemporalConstraintGuardrail' = None) -> ExecutionTrace:
    """Get or create an execution trace for a thread (shared across agents)."""
    if thread_id not in _traces:
        trace = ExecutionTrace(thread_id=thread_id)
        if guardrail:
            trace.choice_constraints = guardrail.choice_constraints  # Direct assignment
        _traces[thread_id] = trace
        logger.debug(f"Created new trace for thread {thread_id}")
    return _traces[thread_id]


class TemporalConstraintGuardrail:
    """Hard guardrail enforcing DECLARE temporal constraints."""
    
    def __init__(self, constraints: List[TemporalConstraint]):
        self.constraints = constraints
        self.name = "temporal_order_constraints"
        self.version = "v1"
        
        
        # Organize constraints by type for efficient checking
        self.response_constraints = []
        self.precedence_constraints = []
        self.chain_constraints = []
        self.choice_constraints = []
        self.coexistence_constraints = []
        self.responded_existence_constraints = []
        
        self.response_map: Dict[str, str] = {}
        self.reverse_response_map: Dict[str, List[str]] = defaultdict(list)

        self.alternate_response_map: Dict[str, str] = {}
        self.reverse_alternate_response_map: Dict[str, List[str]] = defaultdict(list)
        
        self.responded_existence_map: Dict[str, str] = {}
        self.reverse_responded_existence_map: Dict[str, List[str]] = defaultdict(list)

        self.coexistence_map: Dict[str, str] = {}
        self.reverse_coexistence_map: Dict[str, List[str]] = defaultdict(list)

        for c in constraints:
            if c.constraint_type in [ConstraintType.RESPONSE]:
                self.response_constraints.append(c)
                self.response_map[c.antecedent] = c.consequent
                self.reverse_response_map[c.consequent].append(c.antecedent)
            elif c.constraint_type in [ConstraintType.ALTERNATE_RESPONSE]:
                self.alternate_response_constraints.append(c)
                self.alternate_response_map[c.antecedent] = c.consequent
                self.reverse_alternate_response_map[c.consequent].append(c.antecedent)
            elif c.constraint_type in [ConstraintType.PRECEDENCE, ConstraintType.ALTERNATE_PRECEDENCE]:
                self.precedence_constraints.append(c)
            elif c.constraint_type in [ConstraintType.CHAIN_RESPONSE, ConstraintType.CHAIN_PRECEDENCE]:
                self.chain_constraints.append(c)
            elif c.constraint_type in [ConstraintType.CHOICE, ConstraintType.EXCLUSIVE_CHOICE]:
                self.choice_constraints.append(c)
            elif c.constraint_type in [ConstraintType.COEXISTENCE]:
                self.coexistence_constraints.append(c)
                self.coexistence_map[c.antecedent] = c.consequent
                self.coexistence_map[c.consequent] = c.antecedent
                self.reverse_coexistence_map[c.consequent].append(c.antecedent)
                self.reverse_coexistence_map[c.antecedent].append(c.consequent)
            elif c.constraint_type == ConstraintType.RESPONDED_EXISTENCE:
                self.responded_existence_constraints.append(c)
                self.responded_existence_map[c.antecedent] = c.consequent
                self.reverse_responded_existence_map[c.consequent].append(c.antecedent)
        
        # Collect business tools that have constraints (for enforcement)
        self.enforced_tools = set()
        for c in constraints:
            self.enforced_tools.add(c.antecedent)
            self.enforced_tools.add(c.consequent)
            if c.additional_activities:
                self.enforced_tools.update(c.additional_activities)
        
        # For backwards compatibility - tools property returns enforced_tools
        self.tools = list(self.enforced_tools)
        
        logger.debug(f"Temporal guardrail initialized with {len(constraints)} DECLARE constraints")
        logger.debug(f"  Will ENFORCE constraints on: {sorted(self.enforced_tools)}")

    def applies_to(self, tool_name: str) -> bool:
        """Check if this guardrail applies to the tool."""
        return True
    
    def evaluate(self, context: GuardrailContext, thread_id: str) -> Verdict:
        """Evaluate all temporal constraints."""
        
        tool_name = context.tool_name
        tool_args = context.tool_args
        tool_call_id = getattr(context, 'tool_call_id', 'unknown')
        
        # Get the shared trace for this thread
        trace = get_or_create_trace(thread_id, guardrail=self)
        
        # ========== STEP 1: Check ALL constraints that could DENY ==========
        
        # Check ALTERNATE RESPONSE (can deny immediately)
        if tool_name in self.alternate_response_map:
            expected_consequent = self.alternate_response_map[tool_name]
            pending_responses = trace.get_unfulfilled_responses()
            if tool_name in pending_responses:
                return Verdict(
                    effect=Effect.DENY,
                    guardrail_name=self.name,
                    guardrail_type="temporal_hard",
                    reason_internal=f"AlternateResponse violation: {tool_name} occurred again before {expected_consequent} was fulfilled",
                    reason_for_llm=f"❌ Alternate Response violation. {tool_name} cannot occur again until {expected_consequent} has occurred."
                )
        
        choice_made = False
        choice_constraint = None
        
        for constraint in self.constraints:
            # Skip constraints that are handled differently (tracking, not denying)
            if constraint.constraint_type in [
                ConstraintType.RESPONSE,
                ConstraintType.ALTERNATE_RESPONSE,
                ConstraintType.RESPONDED_EXISTENCE,
                ConstraintType.COEXISTENCE
            ]:
                continue  # These are tracked, not denied immediately
            
            # Check if this tool is involved in this constraint
            if tool_name in [constraint.antecedent, constraint.consequent]:
                verdict = self._check_constraint(constraint, trace, tool_name)
                if verdict and verdict.effect == Effect.DENY:
                    return verdict
                
                # Track if this was a choice (for logging later)
                if constraint.constraint_type in [ConstraintType.CHOICE, ConstraintType.EXCLUSIVE_CHOICE]:
                    all_choice_tools = [constraint.antecedent, constraint.consequent]
                    if constraint.additional_activities:
                        all_choice_tools.extend(constraint.additional_activities)
                    if not any(trace.has_occurred(tool) for tool in all_choice_tools):
                        choice_made = True
                        choice_constraint = constraint
        
        # ========== STEP 2: Only allow if all constraints pass ==========
        
        # Record the event
        trace.add_event(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            timestamp=datetime.now(),
            context=tool_args
        )

        if choice_made and choice_constraint:
            self._log_choice_made(choice_constraint, trace, tool_name)
        
        # ========== STEP 3: Track and fulfill responses ==========
        
        # Track RESPONSE
        if tool_name in self.response_map:
            expected_consequent = self.response_map[tool_name]
            trace.add_pending_response(
                antecedent=tool_name,
                consequent=expected_consequent,
                timestamp=datetime.now(),
                tool_call_id=tool_call_id,
                constraint_type="RESPONSE"
            )
            logger.debug(f"Response tracking: {tool_name} occurred, expecting {expected_consequent} to happen later")
        
        # Track ALTERNATE RESPONSE
        if tool_name in self.alternate_response_map:
            expected_consequent = self.alternate_response_map[tool_name]
            trace.add_pending_response(
                antecedent=tool_name,
                consequent=expected_consequent,
                timestamp=datetime.now(),
                tool_call_id=tool_call_id,
                constraint_type="ALTERNATE_RESPONSE"
            )
            logger.debug(f"AlternateResponse tracking: {tool_name} occurred, expecting {expected_consequent} to happen later")
        
        # Track RESPONDED_EXISTENCE
        if tool_name in self.responded_existence_map:
            expected_consequent = self.responded_existence_map[tool_name]
            if not trace.has_occurred(expected_consequent):
                trace.add_pending_response(
                    antecedent=tool_name,
                    consequent=expected_consequent,
                    timestamp=datetime.now(),
                    tool_call_id=tool_call_id,
                    constraint_type="RESPONDED_EXISTENCE"
                )
                logger.debug(f"Responded Existence tracking: {tool_name} occurred, expecting {expected_consequent} to happen later")
            else:
                logger.debug(f"Responded Existence fulfilled: {tool_name} occurred and {expected_consequent} has already occurred")
        
        # Track COEXISTENCE
        if tool_name in self.coexistence_map:
            expected_consequent = self.coexistence_map[tool_name]
            if not trace.has_occurred(expected_consequent):
                trace.add_pending_response(
                    antecedent=tool_name,
                    consequent=expected_consequent,
                    timestamp=datetime.now(),
                    tool_call_id=tool_call_id,
                    constraint_type="COEXISTENCE"
                )
                logger.debug(f"Coexistence tracking: {tool_name} occurred, expecting {expected_consequent} to happen later")
        
        # Fulfill pending responses
        if tool_name in self.reverse_response_map:
            fulfilled = trace.fulfill_response(
                consequent=tool_name,
                timestamp=datetime.now(),
                tool_call_id=tool_call_id
            )
            if fulfilled:
                logger.debug(f"Response fulfilled: {tool_name} occurred and {fulfilled['antecedent']} has already occurred")
        
        if tool_name in self.reverse_alternate_response_map:
            fulfilled = trace.fulfill_response(
                consequent=tool_name,
                timestamp=datetime.now(),
                tool_call_id=tool_call_id
            )
            if fulfilled:
                logger.debug(f"Alternate Response fulfilled: {tool_name} occurred and {fulfilled['antecedent']} has already occurred")
        
        if tool_name in self.reverse_responded_existence_map:
            fulfilled = trace.fulfill_response(
                consequent=tool_name,
                timestamp=datetime.now(),
                tool_call_id=tool_call_id
            )
            if fulfilled:
                logger.debug(f"Responded Existence fulfilled: {tool_name} occurred and {fulfilled['antecedent']} has already occurred")
        
        if tool_name in self.reverse_coexistence_map:
            fulfilled = trace.fulfill_response(
                consequent=tool_name,
                timestamp=datetime.now(),
                tool_call_id=tool_call_id
            )
            if fulfilled:
                logger.debug(f"Coexistence fulfilled: {tool_name} occurred and {fulfilled['antecedent']} has already occurred")
        
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type="temporal_hard",
            reason_internal=f"Temporal constraints satisfied for {tool_name}"
        )
    
    def get_response_status(self, thread_id: str) -> Dict:
        """Get current response tracking status for a thread."""
        if thread_id not in _traces:
            return {"status": "no_trace", "pending_responses": []}
        
        trace = _traces[thread_id]
        pending = trace.get_unfulfilled_responses()
        
        return {
            "status": "active" if trace.session_active else "ended",
            "pending_responses": [
                {
                    "antecedent": antecedent,
                    "consequent": pending_list[0]['consequent'],
                    "occurred_at": pending_list[0]['timestamp'].isoformat(),
                    "waiting_seconds": (datetime.now() - pending_list[0]['timestamp']).total_seconds()
                }
                for antecedent, pending_list in pending.items()
            ],
            "completed_responses": trace.completed_responses
        }
    
    def _check_constraint(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_tool: str) -> Optional[Verdict]:
        """Check a specific DECLARE constraint."""
        
        if constraint.constraint_type in [ConstraintType.PRECEDENCE, ConstraintType.ALTERNATE_PRECEDENCE]:
            return self._check_precedence(constraint, trace, current_tool)
        
        elif constraint.constraint_type == ConstraintType.CHAIN_RESPONSE:
            return self._check_chain_response(constraint, trace, current_tool)
        
        elif constraint.constraint_type == ConstraintType.CHAIN_PRECEDENCE:
            return self._check_chain_precedence(constraint, trace, current_tool)
        
        elif constraint.constraint_type == ConstraintType.CHOICE or constraint.constraint_type == ConstraintType.EXCLUSIVE_CHOICE:
            return self._check_choice_constraint(constraint, trace, current_tool)
        

        return None
    
    def _check_precedence(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_tool: str) -> Optional[Verdict]:
        """Check precedence constraint: ¬B W A (B cannot occur until A has occurred)."""
        
        # Only check when the consequent (B) occurs
        if current_tool != constraint.consequent:
            return None
        
        # Check if A has occurred before this B
        for event in reversed(trace.events[:-1]):  # Exclude current event
            
            if event['tool_name'] == constraint.antecedent:
                # Check if this is ALTERNATE PRECEDENCE: no B between A and this B
                if constraint.constraint_type == ConstraintType.ALTERNATE_PRECEDENCE:
                    b_between = any(
                        e['tool_name'] == constraint.consequent 
                        for e in trace.get_events_between(constraint.antecedent, constraint.consequent)
                    )
                    if b_between:
                        return Verdict(
                            effect=Effect.DENY,
                            guardrail_name=self.name,
                            guardrail_type="temporal_hard",
                            reason_internal=f"AlternatePrecedence violation: Another {constraint.consequent} occurred between {constraint.antecedent} and this {constraint.consequent}",
                            reason_for_llm=f"❌ Alternate Precedence violation. No other {constraint.consequent} can occur between {constraint.antecedent} and {constraint.consequent}."
                        )
                
                logger.debug(f"Precedence satisfied: {constraint.antecedent} occurred before {constraint.consequent}")
                return None
        
        # No A found before B
        antecedent_occurred = any(
            e['tool_name'] == constraint.antecedent 
            for e in trace.events 
        )
        
        if not antecedent_occurred:
            return Verdict(
                effect=Effect.DENY,
                guardrail_name=self.name,
                guardrail_type="temporal_hard",
                reason_internal=f"Precedence violation: {constraint.antecedent} must precede {constraint.consequent}",
                reason_for_llm=f"❌ Missing prerequisite. {constraint.antecedent} must happen before {constraint.consequent}."
            )
        
        return None
    
    def _check_chain_response(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_tool: str) -> Optional[Verdict]:
        """Check chain response: G(A → X(B)). A must be immediately followed by B."""
        
        # Check if this is the consequent (B)
        if current_tool != constraint.consequent:
            return None
        
        # Check if the immediate predecessor is A (or admin tools)
        business_events = [e for e in trace.events if e['tool_name']]
        
        if len(business_events) < 1:
            return Verdict(
                effect=Effect.DENY,
                guardrail_name=self.name,
                guardrail_type="temporal_hard",
                reason_internal=f"ChainResponse violation: {constraint.antecedent} must immediately precede {constraint.consequent}, but no preceding event found",
                reason_for_llm=f"❌ {constraint.antecedent} must immediately precede {constraint.consequent}."
            )
        
        # Get the immediate predecessor business event
        predecessor = business_events[-1]
        
        if predecessor['tool_name'] != constraint.antecedent:
            return Verdict(
                effect=Effect.DENY,
                guardrail_name=self.name,
                guardrail_type="temporal_hard",
                reason_internal=f"ChainResponse violation: Expected {constraint.antecedent} immediately before {constraint.consequent}, but found {predecessor['tool_name']}",
                reason_for_llm=f"❌ {constraint.antecedent} must immediately precede {constraint.consequent}. (Found {predecessor['tool_name']} instead)"
            )
        
        logger.debug(f"ChainResponse satisfied: {constraint.antecedent} → {constraint.consequent} immediately")
        return None
    
    def _check_chain_precedence(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_tool: str) -> Optional[Verdict]:
        """Check chain precedence: G(X(B) → A) ∧ ¬B. Every B is immediately preceded by A."""
        
        # When B occurs, check that A is immediately before
        if current_tool == constraint.consequent:
            business_events = [e for e in trace.events if e['tool_name']]
            
            if len(business_events) < 1:
                return Verdict(
                    effect=Effect.DENY,
                    guardrail_name=self.name,
                    guardrail_type="temporal_hard",
                    reason_internal=f"ChainPrecedence violation: {constraint.consequent} must be immediately preceded by {constraint.antecedent}, but no preceding event found",
                    reason_for_llm=f"❌ {constraint.consequent} must be immediately preceded by {constraint.antecedent}."
                )
            
            predecessor = business_events[-1]
            if predecessor['tool_name'] != constraint.antecedent:
                return Verdict(
                    effect=Effect.DENY,
                    guardrail_name=self.name,
                    guardrail_type="temporal_hard",
                    reason_internal=f"ChainPrecedence violation: Expected {constraint.antecedent} immediately before {constraint.consequent}, but found {predecessor['tool_name']}",
                    reason_for_llm=f"❌ {constraint.antecedent} must immediately precede {constraint.consequent}. (Found {predecessor['tool_name']} instead)"
                )
            
            logger.debug(f"ChainPrecedence satisfied: {constraint.antecedent} → {constraint.consequent} immediately")
            return None
        
        return None
    
    def _check_choice_constraint(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_tool: str) -> Optional[Verdict]:
        """
        Validate choice constraint without logging.
        """
        all_choice_tools = [constraint.antecedent, constraint.consequent]
        if constraint.additional_activities:
            all_choice_tools.extend(constraint.additional_activities)
        
        # Check if any of the choice tools have already occurred (excluding current tool)
        occurred_others = [tool for tool in all_choice_tools 
                        if tool != current_tool and trace.has_occurred(tool)]
        
        # EXCLUSIVE_CHOICE: check if any other tool already occurred
        if constraint.constraint_type == ConstraintType.EXCLUSIVE_CHOICE:
            if occurred_others:
                return Verdict(
                    effect=Effect.DENY,
                    guardrail_name=self.name,
                    guardrail_type="temporal_hard",
                    reason_internal=f"Exclusive choice violation: Cannot have both {', '.join(all_choice_tools)}",
                    reason_for_llm=f"❌ Cannot use both {', '.join(all_choice_tools)}. Choose one."
                )
            # Valid - but don't log yet
            return None
        
        # CHOICE (inclusive OR): always valid (no violation possible)
        # CHOICE only fails at session end if neither was called
        return None

    def _log_choice_made(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_tool: str):
        """Log that a choice was made (called after all constraints pass)."""
        all_choice_tools = [constraint.antecedent, constraint.consequent]
        if constraint.additional_activities:
            all_choice_tools.extend(constraint.additional_activities)
        
        if constraint.constraint_type == ConstraintType.EXCLUSIVE_CHOICE:
            logger.info(f"Exclusive choice made: {current_tool} selected from options: {', '.join(all_choice_tools)}")
        else:
            logger.info(f"Choice made: {current_tool} selected from options: {', '.join(all_choice_tools)}")
    


def create_temporal_guardrail(config_path: str) -> TemporalConstraintGuardrail:
    """Factory to create temporal guardrail from YAML config."""
    
    import yaml
    from pathlib import Path
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    constraints = []
    
    for c_data in data.get('temporal_constraints', []):
        constraint = TemporalConstraint.from_dict(c_data)
        constraints.append(constraint)
    
    logger.info(f"Loaded {len(constraints)} DECLARE constraints from {config_path}")
    
    type_counts = defaultdict(int)
    for c in constraints:
        type_counts[c.constraint_type.value] += 1
    
    logger.info(f"Constraint types: {dict(type_counts)}")
    
    return TemporalConstraintGuardrail(constraints)