# Coffee Shop Order Process

Lanes (agents): order_agent, inventory_agent, barista_agent, customer_service_agent.

## Activities

Each row is one activity. The supervisor MUST refer to activities by their `id`
(A01..A10/A05b/A09b), never by any other label.

| id   | slug                       | agent                  | trigger    | tool             | terminal | follows                |
|------|----------------------------|------------------------|------------|------------------|----------|------------------------|
| A01  | identify_customer_request  | order_agent            | message    | —                | no       | (start)                |
| A02  | create_order               | order_agent            | tool_call  | process_order    | no       | A01                    |
| A03  | check_stock                | inventory_agent        | tool_call  | check_inventory  | no       | A02                    |
| A04  | place_food_on_tray         | inventory_agent        | tool_call  | place_on_tray    | no       | A03 (AND-split branch) |
| A05  | brew_coffee                | barista_agent          | tool_call  | start_preparation| no       | A03 (AND-split branch) |
| A05b | brew_coffee                | barista_agent          | tool_call  | end_preparation  | no       | A05                    |
| A06  | purchase_order             | order_agent            | tool_call  | calculate_total  | no       | A03 (AND-split branch) |
| A07  | handout_order              | order_agent            | message    | —                | YES      | A04 ∧ A05b ∧ A06 (AND-join) |
| A08  | investigate_complaint      | customer_service_agent | message    | —                | no       | A01 (XOR alt branch)   |
| A09  | offer_refund               | customer_service_agent | tool_call  | offer_refund     | YES      | A08 (valid claim)      |
| A09b | offer_refund               | customer_service_agent | tool_call  | offer_partial_refund | YES  | A08 (valid claim)      |
| A10  | reject_complaint           | customer_service_agent | message    | —                | YES      | A08 (default)          |

## Control flow

- **Start** → A01.
- After A01, **XOR**: order branch (→ A02) OR complaint branch (→ A08).
- Order branch: A02 → A03 → AND-split { A04, A05→A05b, A06 } → AND-join → A07 (terminal).
- Complaint branch: A08 → XOR { valid claim → A09 or A09b (terminal); default → A10 (terminal) }.

## Cross-lane handoffs

A handoff (`transfer_to_agent` tool call) is NOT itself an activity. It terminates the
**most recent open activity in the source agent's lane** with reason
`via_handoff_to_<target_agent>`. Typical handoffs:
- order_agent → inventory_agent (after A02).
- inventory_agent → barista_agent (after A04 or A03; whichever is most recent).
- any lane → customer_service_agent (escalation).

## Ambiguity rules for `(agent, trigger=message, tool=None)` events

- order_agent: A01 if no AND-split branch (A04/A05/A05b/A06) has fired yet, else A07.
- customer_service_agent: A08 if no prior A08 in log, else A10.
- inventory_agent and barista_agent have NO message-trigger activity → Violation.

## Tool ownership (any other binding is a Violation)

- process_order, calculate_total → order_agent
- check_inventory, place_on_tray → inventory_agent
- start_preparation, end_preparation → barista_agent
- offer_refund, offer_partial_refund → customer_service_agent
- transfer_to_agent → handoff (special case, see above)
