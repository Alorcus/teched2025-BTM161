"""
Temporal order constraints based on Direct Encoding of Declare Constraints in ASP (Chiariello et al., 2025).

Supports all constraint patterns:
- RespondedExistence(A,B): F(A) -> F(B) (if A occurs, B must occur)
- Coexistence(A,B): RespondedExistence(A,B) ∧ RespondedExistence(B,A) (if A occurs, B must occur and vice versa)
- Choice(A,B): F(A ∨ B) (inclusive OR - at least one)
- ExclusiveChoice(A,B): Choice(A,B) ∧ ¬(F(A) ∧ F(B)) (exclusive OR - exactly one)
- Response(A,B): G(A → F(B)) (every A is eventually followed by a B)
- Precedence(A,B): ¬B W A (B cannot occur until A has occurred)
- AlternateResponse(A,B): G(A → X(¬A U B)) (every A is eventually followed by a B, no other A in between)
- AlternatePrecedence(A,B): Precedence(A,B) ∧ G(B → X(Precedence(A,B))) (Every B is preceded by an A without another B in between)
- ChainResponse(A,B): G(A → X(B)) (every A is immediately followed by a B)
- ChainPrecedence(A,B): G(X(B) → A) ∧ ¬B (Every B is immediately preceded by an A, no B without a preceding A)

Note: All constraints are binary (two activities). Multi-activity chains combine multiple binary constraints.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstraintType(str, Enum):
    """DECLARE constraint types as defined in Chiariello et al. (2025)."""
    
    # Existence & Choice constraints
    CHOICE = "choice"                   # F(A ∨ B) - inclusive OR
    EXCLUSIVE_CHOICE = "exclusive_choice"  # Choice ∧ ¬(F(A) ∧ F(B)) - exactly one
    RESPONDED_EXISTENCE = "responded_existence"  # F(A) → F(B)
    COEXISTENCE = "coexistence"         # RespondedExistence(A,B) ∧ RespondedExistence(B,A)
    
    # Response-based constraints
    RESPONSE = "response"               # G(A → F(B))
    ALTERNATE_RESPONSE = "alternate_response"  # G(A → X(¬A U B))
    CHAIN_RESPONSE = "chain_response"   # G(A → X(B))
    
    # Precedence-based constraints
    PRECEDENCE = "precedence"           # ¬B W A
    ALTERNATE_PRECEDENCE = "alternate_precedence"  # Precedence(A,B) ∧ G(B → X(Precedence(A,B)))
    CHAIN_PRECEDENCE = "chain_precedence"  # G(X(B) → A) ∧ ¬B


@dataclass
class TemporalConstraint:
    """
    Represents a DECLARE temporal constraint between two activities.
    
    All core DECLARE constraints are binary (relating exactly two activities).
    Multi-activity chains are represented by combining binary constraints.
    """
    
    id: str
    constraint_type: ConstraintType
    antecedent: str  # First activity (e.g., A in Response(A,B))
    consequent: str  # Second activity (e.g., B in Response(A,B))
    
    # Optional additional activities for complex constraints
    additional_activities: Optional[List[str]] = None
    
    # Metadata
    description: str = ""
    version: str = "v1"
    
    # LTL_f formula (for reference)
    ltl_formula: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalConstraint':
        """Create constraint from dictionary (e.g., from YAML)."""
        return cls(
            id=data['id'],
            constraint_type=ConstraintType(data['constraint_type']),
            antecedent=data['antecedent'],
            consequent=data['consequent'],
            additional_activities=data.get('additional_activities'),
            description=data.get('description', ''),
            version=data.get('version', 'v1'),
            ltl_formula=data.get('ltl_formula'),
        )
    
    def get_ltl_formula(self) -> Optional[str]:
        """Return the LTL_f formula for this constraint if available."""
        if self.ltl_formula:
            return self.ltl_formula
        
        # Generate from type if known
        A = self.antecedent
        B = self.consequent
        
        formulas = {
            ConstraintType.CHOICE: f"F({A} ∨ {B})",
            ConstraintType.EXCLUSIVE_CHOICE: f"F({A} ∨ {B}) ∧ ¬(F({A}) ∧ F({B}))",
            ConstraintType.RESPONDED_EXISTENCE: f"F({A}) → F({B})",
            ConstraintType.COERXISTENCE: f"(F({A}) → F({B})) ∧ (F({B}) → F({A}))",
            ConstraintType.RESPONSE: f"G({A} → F({B}))",
            ConstraintType.PRECEDENCE: f"¬{B} W {A}",
            ConstraintType.ALTERNATE_RESPONSE: f"G({A} → X(¬{A} U {B}))",
            ConstraintType.ALTERNATE_PRECEDENCE: f"Precedence({A},{B}) ∧ G({B} → X(Precedence({A},{B})))",
            ConstraintType.CHAIN_RESPONSE: f"G({A} → X({B}))",
            ConstraintType.CHAIN_PRECEDENCE: f"G(X({B}) → {A}) ∧ ¬{B}",
        }
        
        return formulas.get(self.constraint_type)
    
    def is_response_based(self) -> bool:
        """Check if this is a Response-based constraint."""
        return self.constraint_type in [
            ConstraintType.RESPONSE,
            ConstraintType.ALTERNATE_RESPONSE,
            ConstraintType.CHAIN_RESPONSE,
            ConstraintType.RESPONDED_EXISTENCE,
        ]
    
    def is_precedence_based(self) -> bool:
        """Check if this is a Precedence-based constraint."""
        return self.constraint_type in [
            ConstraintType.PRECEDENCE,
            ConstraintType.ALTERNATE_PRECEDENCE,
            ConstraintType.CHAIN_PRECEDENCE,
        ]
    
    
    def requires_immediate_order(self) -> bool:
        """Check if this constraint requires immediate succession (Chain level)."""
        return self.constraint_type in [
            ConstraintType.CHAIN_RESPONSE,
            ConstraintType.CHAIN_PRECEDENCE,
        ]
    
    def requires_alternate_order(self) -> bool:
        """Check if this constraint requires alternation (Alternate level)."""
        return self.constraint_type in [
            ConstraintType.ALTERNATE_RESPONSE,
            ConstraintType.ALTERNATE_PRECEDENCE,
        ]