# Coffee Shop Order Process — Structured Flow

Source: `docs/order-process-w-compliant.bpmn`
Pool: **Coffee Shop**

## Lanes (Performers)

| Lane | Agent |
|------|-------|
| Lane_08nhbsk | Order Agent |
| Lane_0o0ktpn | Inventory Agent |
| Lane_1rkg2ab | Barista Agent |
| Lane_1yu54m5 | Customer Service Agent |

## Activities

| ID | Activity | Performer |
|----|----------|-----------|
| Activity_1xmfjbf | Identify Customer Request | Order Agent |
| Activity_16794om | Create Order | Order Agent |
| Activity_03ayqb0 | Purchase Order | Order Agent |
| Activity_1squ609 | Handout Order | Order Agent |
| Activity_0mub8wc | Check Stock | Inventory Agent |
| Activity_14zkmol | Place Food on Tray | Inventory Agent |
| Activity_0g8d5ah | Brew Coffee | Barista Agent |
| Activity_0n3u8vo | Investigate into Customer Complaint | Customer Service Agent |
| Activity_1079o4t | Offer Refund | Customer Service Agent |
| Activity_1cywvq2 | Reject Complaint | Customer Service Agent |

## Events & Gateways

| ID | Type | Role |
|----|------|------|
| StartEvent_1waz7sk | Start Event | Process entry (Order Agent lane) |
| Gateway_148j4ih | Exclusive Gateway (XOR) | Order vs. complaint branch |
| Gateway_0im3hsr | Parallel Gateway (AND-split) | Fan out to 3 concurrent branches |
| Gateway_0xm5qyk | Parallel Gateway (AND-join) | Synchronize the 3 branches |
| Gateway_1ou88ca | Exclusive Gateway (XOR, default=Reject) | Valid-claim decision |
| Event_0x9r06d | End Event | Successful order completion (Order Agent lane) |
| Event_0n7u6j4 | End Event | Refund issued (Customer Service Agent lane) |
| Event_14cgp1c | End Event | Complaint rejected (Customer Service Agent lane) |

## Control Flow (Sequence)

```
Start (Order Agent)
  └─> Identify Customer Request          [Order Agent]
        └─> XOR (Gateway_148j4ih)
              ├─> Create Order           [Order Agent]                                  ← normal order branch
              │     └─> Check Stock      [Inventory Agent]
              │           └─> AND-split (Gateway_0im3hsr)
              │                 ├─> Place Food on Tray [Inventory Agent] ─┐
              │                 ├─> Brew Coffee        [Barista Agent]    ├─> AND-join (Gateway_0xm5qyk)
              │                 └─> Purchase Order     [Order Agent]      ─┘
              │                                                                  └─> Handout Order  [Order Agent]
              │                                                                        └─> End (Event_0x9r06d)
              │
              └─> Investigate into Customer Complaint  [Customer Service Agent]    ← complaint branch
                    └─> XOR (Gateway_1ou88ca, default = Reject)
                          ├─> Offer Refund            [Customer Service Agent] ─> End (Event_0n7u6j4)    [valid claim]
                          └─> Reject Complaint        [Customer Service Agent] ─> End (Event_14cgp1c)    [default]
```

## Sequence Flows (Edges)

| Flow ID | From | To | Note |
|---------|------|----|------|
| Flow_0lpa080 | StartEvent_1waz7sk | Identify Customer Request | |
| Flow_1liilvc | Identify Customer Request | Gateway_148j4ih (XOR) | |
| Flow_0qw9jmz | Gateway_148j4ih | Create Order | normal-order branch |
| Flow_11zrbf5 | Gateway_148j4ih | Investigate into Customer Complaint | complaint branch |
| Flow_0g1rsxf | Create Order | Check Stock | |
| Flow_0ccjnra | Check Stock | Gateway_0im3hsr (AND-split) | |
| Flow_03bfkb0 | Gateway_0im3hsr | Place Food on Tray | |
| Flow_00qkszu | Gateway_0im3hsr | Brew Coffee | |
| Flow_07e0kpj | Gateway_0im3hsr | Purchase Order | |
| Flow_0ja3o6y | Place Food on Tray | Gateway_0xm5qyk (AND-join) | |
| Flow_1c1f5s9 | Brew Coffee | Gateway_0xm5qyk (AND-join) | |
| Flow_0unrso1 | Purchase Order | Gateway_0xm5qyk (AND-join) | |
| Flow_0sjg4rv | Gateway_0xm5qyk | Handout Order | |
| Flow_0ux09ym | Handout Order | Event_0x9r06d (End) | |
| Flow_0cxwa46 | Investigate into Customer Complaint | Gateway_1ou88ca | |
| Flow_0dleb6u | Gateway_1ou88ca | Offer Refund | label: "valid claim" |
| Flow_1ocobik | Gateway_1ou88ca | Reject Complaint | default branch |
| Flow_1ivbibj | Offer Refund | Event_0n7u6j4 (End) | |
| Flow_0eyiqum | Reject Complaint | Event_14cgp1c (End) | |

## Control-Flow Semantics

- **Sequential prefix:** Start → Identify Customer Request.
- **First decision (XOR Gateway_148j4ih):** customer request → either continue with order processing (Create Order …) or escalate to the Customer Service lane (Investigate into Customer Complaint …). Exactly one branch is taken.
- **Order-processing branch:**
  - **Sequential:** Create Order → Check Stock.
  - **Parallel block:** AND-split spawns three concurrent branches:
    1. Inventory Agent: Place Food on Tray
    2. Barista Agent: Brew Coffee
    3. Order Agent: Purchase Order
  - **Synchronization:** AND-join waits for all three branches.
  - **Sequential suffix:** Handout Order → End (Event_0x9r06d).
- **Complaint branch (Customer Service Agent lane):**
  - Investigate into Customer Complaint → XOR (Gateway_1ou88ca).
  - Branches: `valid claim` → Offer Refund → End (Event_0n7u6j4); default → Reject Complaint → End (Event_14cgp1c).

## Cross-Lane Handoffs

1. Order Agent → Inventory Agent: after *Create Order*, control passes to *Check Stock*.
2. Inventory Agent → (Inventory Agent + Barista Agent + Order Agent): AND-split distributes work across all three lanes.
3. (All three lanes) → Order Agent: AND-join returns control to Order Agent for *Handout Order*.
4. Order Agent → Customer Service Agent: when XOR Gateway_148j4ih takes the complaint branch, control passes from *Identify Customer Request* to *Investigate into Customer Complaint*.

The complaint branch terminates entirely within the Customer Service Agent lane (no cross-lane handoff after the investigation).
