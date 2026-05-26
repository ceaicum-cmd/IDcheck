# Social Hierarchy Engine

## Purpose

Tracks rank, exposure, humiliation pressure, elite attention, social debt, and reputation volatility.

## Core Variables

```json
{
  "rank_level": 0,
  "rank_label": "Unranked",
  "elite_attention": 0,
  "humiliation_exposure": 0,
  "reputation_stability": 50,
  "social_debt": 0,
  "house_protection": null,
  "active_demotions": [],
  "active_privileges": []
}
```

## Rank Bands

| Level | Label | Meaning |
|---|---|---|
| 0 | Unseen | no status |
| 1 | Amusement | present for elite entertainment |
| 2 | Tolerated | allowed to remain |
| 3 | Tested | under active evaluation |
| 4 | Useful | has situational value |
| 5 | Recognized | has temporary social identity |
| 6 | Protected | shielded by a House or patron |

## Status Events

- invitation received
- protocol violated
- silence endured
- public correction survived
- elite laughter triggered
- symbolic object accepted
- House debt incurred
- patron protection granted

## Engine Rule

The user does not gain rank through confidence alone. Rank is earned by surviving observation without collapsing into need, panic, vulgarity, or uncontrolled reaction.
