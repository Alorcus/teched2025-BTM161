"""Temporal order constraints based on BPMN2Constraints (Bergman et al., 2023).

Supports constraint patterns:
- Sequence: A must be followed by B
- Parallel: A and B can happen in any order
- Choice: Either A or B (but not both)
- Loop: Repeat A until condition
- Time-based: A must happen within time T after B
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re


class ConstraintType(str, Enum):
    SEQUENCE = "sequence"           # A then B (strict order)
    PARALLEL = "parallel"           # A and B (no order constraint)
    CHOICE = "choice"               # A or B (exclusive)
    LOOP = "loop"                   # Repeat A
    TIME_BOUNDED = "time_bounded"   # A within T of B
    PRECEDENCE = "precedence"       # A must occur before B (not necessarily immediately)
    RESPONSE = "response"           # B must occur after A
    CHAIN_RESPONSE = "chain_response"  # A then B then C


class TemporalOperator(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    BETWEEN = "between"
    WITHIN = "within"


@dataclass
class TemporalConstraint:
    """Represents a temporal constraint between activities/events."""
    
    id: str
    constraint_type: ConstraintType
    antecedent: str  # First activity/tool call
    consequent: str  # Second activity/tool call
    time_bound: Optional[timedelta] = None
    description: str = ""
    version: str = "v1"
    
    # For loop constraints
    max_iterations: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalConstraint':
        """Create constraint from dictionary (e.g., from YAML)."""
        time_bound = None
        if data.get('time_bound_seconds'):
            time_bound = timedelta(seconds=data['time_bound_seconds'])
        
        return cls(
            id=data['id'],
            constraint_type=ConstraintType(data['constraint_type']),
            antecedent=data['antecedent'],
            consequent=data['consequent'],
            time_bound=time_bound,
            description=data.get('description', ''),
            version=data.get('version', 'v1'),
            max_iterations=data.get('max_iterations'),
        )


@dataclass 
class ExecutionTrace:
    """Tracks the execution history for constraint checking."""
    
    agent_id: str
    thread_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event(self, tool_name: str, tool_call_id: str, timestamp: datetime, context: Dict):
        """Record a tool execution event."""
        self.events.append({
            'tool_name': tool_name,
            'tool_call_id': tool_call_id,
            'timestamp': timestamp,
            'context': context,
            'order': len(self.events) + 1
        })
    
    def get_occurrence_order(self, tool_name: str) -> List[int]:
        """Get the occurrence order numbers for a tool."""
        return [e['order'] for e in self.events if e['tool_name'] == tool_name]
    
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