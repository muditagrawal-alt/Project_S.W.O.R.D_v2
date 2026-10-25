# Interval labelling guidelines (Phase 5)

These labels train and evaluate the temporal threat model. The file format is defined in `fusion/labels.py`.

## Principles

1. **Label what is visible.** A label describes observable events in the video, never a person's identity,
   appearance, group membership or presumed intent. If you cannot see it, do not label it.
2. **Use the highest applicable level.** When several apply, choose the highest. Overlapping intervals are allowed;
   the highest wins.
3. **Mark boundaries at visible changes.** An interval starts at the first frame where the defining evidence is
   visible and ends at the last such frame. Do not stretch intervals to cover what probably happened off-screen.
4. **Mark uncertainty in `notes`** (e.g. "weapon possibly a phone"). Do not guess a higher level.

## Levels

| Level | Use when | Typical examples | Not this level |
|---|---|---|---|
| NORMAL | nothing threat-relevant is visible | walking, standing, talking, using a phone, carrying bags | — |
| ELEVATED | unusual but not threatening behaviour | running, sneaking or stealthy approach with empty hands, loitering in a marked restricted zone, a weapon-like object lying unattended | jogging for exercise in an open area: NORMAL |
| SUSPICIOUS_ACTIVITY | physical aggression or threatening behaviour **without** a visible weapon | fighting, punching, kicking, pushing someone down, grabbing or dragging a person, threatening gestures at close range | a robbery where a weapon is clearly visible: WEAPON_RELATED_EVENT |
| WEAPON_RELATED_EVENT | a weapon (firearm, knife, blade, blunt weapon used as one) is **visibly** carried, drawn, aimed, brandished or used | carrying a rifle, aiming a handgun, reloading, stabbing, firing | a phone or other object that only looks like a weapon in one blurry frame: note it, use the level of the rest of the scene |

## Scope

- **`scene`** (default): the interval applies to the whole frame.
- **`person`**: the label belongs to one person. Set `anchor` to one instant and that person's box
  (`{"t": 12.4, "bbox": [x1, y1, x2, y2]}`). The box is matched to tracks at training time, so labels survive
  tracker changes (Phase 2 will be replaced).

## Provenance (`source`)

| Value | Meaning |
|---|---|
| `derived:<rule>` | produced automatically from dataset annotations; **unverified** |
| `human:<annotator>` | drawn by a person |
| `human-verified:<annotator>` | a derived interval a person checked and kept, or corrected |

Training and evaluation report which sources were used. A paper result should state the share of human-verified
labels.

## Derived label rules in this repository (`research/tools/labels/derive_intervals.py`)

| Dataset | Rule | Coverage |
|---|---|---|
| Mendeley firearm set (Ruiz-Santaquiteria et al. 2024), gun videos (codes PC\*, PA\*) | WEAPON_RELATED_EVENT from the first to the last frame with a weapon box; gaps < 1 s merged | only that span (the rest of the video is unknown, because the weapon may be occluded) |
| Mendeley No_Gun | N3 running and N4 sneaking up → ELEVATED; all other N codes → NORMAL; whole video | exhaustive |
| UCF-Crime test videos (official temporal annotations) | Shooting → WEAPON_RELATED_EVENT; Robbery → SUSPICIOUS_ACTIVITY (**needs review**: upgrade to WEAPON_RELATED_EVENT if a weapon is visible); Fighting, Assault → SUSPICIOUS_ACTIVITY; outside the annotated segments and Normal videos → NORMAL | exhaustive |
| Mock-attack CCTV (USRT) | WEAPON_RELATED_EVENT where a weapon box is annotated (gaps ≤ 1.25 s merged) | exhaustive; **test only, never for training** |

## Review workflow

1. Open the labelling tool (`python -m service`, then go to http://127.0.0.1:8000/labeler).
2. Pick a video. The derived intervals are pre-loaded.
3. Play through, and for each interval:
   - keep it: mark it human-verified;
   - correct it: adjust boundaries or level, then mark it human-verified;
   - delete it, or add one that is missing.
4. Save. Every save is written to the database with annotator and time, and exported to CSV.
5. Double-label at least 10% of videos with a second annotator. `research/tools/labels/agreement.py` reports
   agreement (frame-level Cohen's κ, boundary offsets).
