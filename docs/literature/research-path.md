# Research path around AROS

## The central geometric thread

1. **Interaction geometry — Zhao et al. (2014).** The interaction bisector surface represents spatial relations between objects through the space between them. It provides the geometric building block for reasoning about interactions. [Source](https://www.research.ed.ac.uk/en/publications/indexing-3d-scenes-using-the-interaction-bisector-surface/).
2. **One-example rigid interactions — Ruiz & Mayol-Cuevas (2018).** The interaction tensor augments bisector-based geometry with provenance vectors, allowing a single example of a rigid object–scene interaction to be matched in new scenes. [Source](https://doi.org/10.1109/ICRA.2018.8462835).
3. **Scalable search — Ruiz & Mayol-Cuevas (2020).** Deep 3D saliency is combined with the interaction tensor to help identify promising points for geometric affordance detection. [Source](https://doi.org/10.3389/fnbot.2020.00045).
4. **Articulated humans — Pacheco-Ortega & Mayol-Cuevas (2023).** AROS uses proximity and additional clearance information to describe articulated human poses in scanned environments. Its paper explicitly identifies the IBS and interaction-tensor line as inspiration. [Source](https://doi.org/10.3389/frobt.2023.1076780).

This is a conceptual and chronological reading path; it should not be read as proof that every paper directly cites every preceding paper.

## Related approaches

| Approach | What it contributes | Relationship to AROS |
| --- | --- | --- |
| Action maps, e.g. SceneGrok | Connect human actions, poses and scene features to locations. | Earlier related line described in AROS. |
| Simulation-based reasoning, e.g. Wu et al. | Tests possible interactions with a simulated articulated human and physical constraints. | Complementary treatment of feasibility; see its own evaluation before comparing results. |
| PLACE | Learns scene-conditioned human body placement. | AROS compares against PLACE and uses its AdvOptim refinement. |
| POSA | Learns body-centred human–scene interaction, including contact. | AROS compares against the contact-only variant using matched input poses. |

Find the papers and original links in the [collection](papers.md). These methods differ in inputs, supervision and outputs; benchmark results are not directly interchangeable without checking their experimental settings.

## Translating the thread into a robotics project

One plausible research direction is to capture an example of a robot interaction, describe the relation between robot geometry and local scene geometry, search for similar relationships in new scenes, and then test candidates using robot-specific kinematics, collision checking and dynamics. This is a proposed adaptation, not a capability established by AROS.

Questions to decide first: What robot and task? Which contact and free-space relationships matter? Can the required scene geometry be observed? What makes an action successful, and what baselines and held-out scenes would give a fair test? Record these decisions before designing the system.
