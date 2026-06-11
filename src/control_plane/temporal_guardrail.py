"""Temporal guardrail evaluator using BPMN2Constraints patterns."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .temporal_constraints import TemporalConstraint, ConstraintType
from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.temporal_guardrail")

# Global trace storage - keyed by thread_id (shared across agents)
_traces: Dict[str, 'ExecutionTrace'] = {}


class ExecutionTrace:
    """Tracks the execution history for constraint checking."""
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.events: List[Dict[str, Any]] = []
    
    def add_event(self, tool_name: str, tool_call_id: str, timestamp: datetime, context: Dict):
        """Record a tool execution event."""
        self.events.append({
            'tool_name': tool_name,
            'tool_call_id': tool_call_id,
            'timestamp': timestamp,
            'context': context,
            'order': len(self.events) + 1
        })
        logger.debug(f"Trace {self.thread_id}: recorded {tool_name} as event {len(self.events)}")
    
    def has_occurred(self, tool_name: str) -> bool:
        """Check if a tool has occurred."""
        return any(e['tool_name'] == tool_name for e in self.events)
    
    def get_last_occurrence(self, tool_name: str) -> Optional[Dict]:
        """Get the last occurrence of a tool."""
        for e in reversed(self.events):
            if e['tool_name'] == tool_name:
                return e
        return None
    
    def get_time_since(self, tool_name: str) -> Optional[timedelta]:
        """Get time since the last occurrence of a tool."""
        last = self.get_last_occurrence(tool_name)
        if last:
            return datetime.now() - last['timestamp']
        return None


def get_or_create_trace(thread_id: str) -> ExecutionTrace:
    """Get or create an execution trace for a thread (shared across agents)."""
    if thread_id not in _traces:
        _traces[thread_id] = ExecutionTrace(thread_id=thread_id)
        logger.debug(f"Created new trace for thread {thread_id}")
    return _traces[thread_id]


class TemporalConstraintGuardrail:
    """Hard guardrail enforcing temporal order constraints."""
    
    def __init__(self, constraints: List[TemporalConstraint]):
        self.constraints = constraints
        self.name = "temporal_order_constraints"
        self.version = "v1"
        # Collect all tools that this guardrail applies to
        self.tools = list(set([c.antecedent for c in constraints] + [c.consequent for c in constraints]))
        logger.info(f"Temporal guardrail initialized with {len(constraints)} constraints for tools: {self.tools}")
    
    def applies_to(self, tool_name: str) -> bool:
        """Check if this guardrail applies to the tool."""
        return tool_name in self.tools
    
    def evaluate(self, context: GuardrailContext, thread_id: str) -> Verdict:
        """Evaluate all temporal constraints."""
        
        tool_name = context.tool_name
        tool_args = context.tool_args
        tool_call_id = getattr(context, 'tool_call_id', 'unknown')
        
        # Get the shared trace for this thread
        trace = get_or_create_trace(thread_id)
        
        # Record this event first
        trace.add_event(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            timestamp=datetime.now(),
            context=tool_args
        )
        
        # Check each constraint that applies to this tool
        for constraint in self.constraints:
            # Check if this tool is the consequent of a constraint
            if tool_name == constraint.consequent:
                verdict = self._check_constraint(constraint, trace, tool_args, thread_id)
                if verdict.effect == Effect.DENY:
                    return verdict
        
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type="temporal_hard",
            reason_internal=f"Temporal constraints satisfied for {tool_name}"
        )
    
    def _check_constraint(self, constraint: TemporalConstraint, trace: ExecutionTrace, current_args: Dict, thread_id: str) -> Verdict:
        """Check a specific constraint."""
        
        antecedent_occurred = trace.has_occurred(constraint.antecedent)
        
        if constraint.constraint_type == ConstraintType.SEQUENCE:
            if not antecedent_occurred:
                return Verdict(
                    effect=Effect.DENY,
                    guardrail_name=self.name,
                    guardrail_type="temporal_hard",
                    reason_internal=f"Sequence violation: {constraint.antecedent} must occur before {constraint.consequent}",
                    reason_for_llm=f"❌ Invalid operation order. You must call {constraint.antecedent} before {constraint.consequent}."
                )
        
        elif constraint.constraint_type == ConstraintType.PRECEDENCE:
            if not antecedent_occurred:
                return Verdict(
                    effect=Effect.DENY,
                    guardrail_name=self.name,
                    guardrail_type="temporal_hard",
                    reason_internal=f"Precedence violation: {constraint.antecedent} must precede {constraint.consequent}",
                    reason_for_llm=f"❌ Missing prerequisite. {constraint.antecedent} must happen before {constraint.consequent}."
                )
        
        elif constraint.constraint_type == ConstraintType.TIME_BOUNDED:
            if antecedent_occurred and constraint.time_bound:
                time_since = trace.get_time_since(constraint.antecedent)
                if time_since and time_since > constraint.time_bound:
                    return Verdict(
                        effect=Effect.DENY,
                        guardrail_name=self.name,
                        guardrail_type="temporal_hard",
                        reason_internal=f"Time bound violation: {constraint.consequent} must occur within {constraint.time_bound.total_seconds()}s of {constraint.antecedent}",
                        reason_for_llm=f"❌ Timeout. {constraint.consequent} must happen within {constraint.time_bound.total_seconds()} seconds of {constraint.antecedent}."
                    )
        
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type="temporal_hard",
            reason_internal=f"Constraint {constraint.id} satisfied"
        )


def create_temporal_guardrail(config_path):
    """Factory to create temporal guardrail from YAML config."""
    
    import yaml
    from pathlib import Path
    from .temporal_constraints import TemporalConstraint, ConstraintType
    
    constraints = []
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    for c_data in data.get('temporal_constraints', []):
        # Handle time_bound_seconds
        time_bound = None
        if c_data.get('time_bound_seconds'):
            time_bound = timedelta(seconds=c_data['time_bound_seconds'])
        
        constraint = TemporalConstraint(
            id=c_data['id'],
            constraint_type=ConstraintType(c_data['constraint_type']),
            antecedent=c_data['antecedent'],
            consequent=c_data['consequent'],
            time_bound=time_bound,
            description=c_data.get('description', ''),
            version=c_data.get('version', 'v1'),
            max_iterations=c_data.get('max_iterations'),
        )
        constraints.append(constraint)
    
    logger.info(f"Loaded {len(constraints)} temporal constraints from {config_path}")
    return TemporalConstraintGuardrail(constraints)