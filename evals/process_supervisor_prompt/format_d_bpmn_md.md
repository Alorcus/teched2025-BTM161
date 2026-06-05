# Coffee Shop Order Process — BPMN structural view

Source: `docs/order-process-w-compliant.bpmn`
Pool: **Coffee Shop**

## Lanes

- Order Agent
- Inventory Agent
- Barista Agent
- Customer Service Agent

## Activities (lane-grouped)

### Order Agent
- A01 — Identify Customer Request
- A02 — Create Order
- A06 — Purchase Order
- A07 — Handout Order   *(end-event activity)*

### Inventory Agent
- A03 — Check Stock
- A04 — Place Food on Tray

### Barista Agent
- A05 — Brew Coffee

### Customer Service Agent
- A08 — Investigate into Customer Complaint
- A09 — Offer Refund   *(end-event activity)*
- A10 — Reject Complaint   *(end-event activity)*

## Events & Gateways

| ID | Type | Role |
|----|------|------|
| Start | Start event | Process entry (Order Agent lane) |
| G1 | XOR (exclusive) gateway | Order vs. complaint branch |
| G2 | AND-split (parallel) gateway | Fan out to A04, A05, A06 |
| G3 | AND-join (parallel) gateway | Synchronize the 3 branches |
| G4 | XOR (exclusive) gateway, default = Reject | Valid-claim decision |
| End-success | End event | Successful order completion |
| End-refund  | End event | Refund issued |
| End-reject  | End event | Complaint rejected |

## Sequence flows

```
Start (Order Agent)
  → A01 Identify Customer Request
  → G1 (XOR)
       ├──(order)──────→ A02 Create Order
       │                    → A03 Check Stock
       │                       → G2 (AND-split)
       │                            ├─→ A04 Place Food on Tray  ──┐
       │                            ├─→ A05 Brew Coffee          ─┤
       │                            └─→ A06 Purchase Order       ─┤
       │                                                          → G3 (AND-join)
       │                                                              → A07 Handout Order
       │                                                                  → End-success
       │
       └──(complaint)──→ A08 Investigate into Customer Complaint
                            → G4 (XOR, default = Reject)
                                 ├──(valid claim)──→ A09 Offer Refund     → End-refund
                                 └──(default)──────→ A10 Reject Complaint → End-reject
```

## Cross-lane handoffs

A `transfer_to_agent` style message, where one lane's agent passes control to
another lane's agent, terminates the most recent open activity in the source
lane (with reason `via_handoff_to_<target_agent>`). Handoffs are not themselves
activities — they correspond to the BPMN sequence-flow edges that cross lane
boundaries.
