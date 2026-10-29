# Vision–language–action models: understanding environments and acting

**Research question:** How can a robot use semantic and temporal understanding of its environment, learned partly from human video, to select and execute actions that generalize to new real-world situations? This is an exploratory review, current to 28 October 2026, not a systematic survey or a definitive ranking.

## The idea in plain language

Your goal appears to be a robot that can interpret a scene, understand what people or objects are doing, infer what actions are possible and useful, and execute those actions in unfamiliar environments. A VLA is one candidate building block: it typically combines camera observations, an instruction and sometimes robot state to predict robot actions. The output can be individual actions or short sequences, repeatedly updated using new observations.

There are four different capabilities to connect:

1. **Semantic understanding:** what objects, instructions and task goals mean, including relationships such as “put the cup inside the cupboard.”
2. **Temporal and physical understanding:** what changed over time, which interaction caused it, and what may happen next.
3. **Affordance and geometry:** where an interaction may be possible, what should contact what, and where free space is needed. This connects directly to the [AROS notes](aros-guide.md).
4. **Embodied control:** converting an intention into motions supported by a particular robot, its cameras, joints, gripper and control interface.

This decomposition is a research framing, not a claim that every system has four separate modules. A language model recognizing an object does not prove it understands physical interactions; a visually plausible generated video does not guarantee a robot can execute it.

## How video contributes

Human videos show interactions in varied environments but usually lack robot control labels. Researchers bridge this gap with latent action representations learned from changes between frames; predicted future frames or world states; 2D contact and tracking cues; or reconstructed 3D geometry and motion. Robot demonstrations, a learned decoder, or another grounding mechanism are still needed to connect such representations to actual controls. The [2026 IJCAI survey](https://ijcai.org/proceedings/2026/871) organizes these routes and discusses the differences in body shape, viewpoints and available labels.

Particularly relevant papers are [UniVLA: Learning to Act Anywhere](https://arxiv.org/abs/2505.06111) for video-derived latent actions; [VIPA-VLA](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_Spatial-Aware_VLA_Pretraining_through_Visual-Physical_Alignment_from_Human_Videos_CVPR_2026_paper.html) for spatial alignment from human video; [CoT-VLA](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html) for imagined visual goals; and [ORION](https://ut-austin-rpl.github.io/ORION-release/) for object relationships and transferring a human demonstration into robot actions.

## The connection to AROS

AROS makes a local interaction explicit: body–scene proximity and clearance predict where a particular pose may fit. A VLA can instead learn to associate observations and instructions with an action sequence, but its internal notion of contact, free space and physical feasibility may be hard to inspect. A promising hypothesis is to combine an explicit representation of **where an interaction can happen** with learned semantic and temporal understanding of **what to do and when**, then ground it in robot demonstrations. The papers collected here do not establish that this combination will outperform a VLA baseline; that requires an experiment.

One possible research question: *Does explicitly supplying contact, object relationships or local geometry improve a video-pretrained robot policy on unseen objects, environments and instructions?* Compare a base VLA with the same VLA plus the proposed representation, and evaluate both on identical held-out cases.

## Suggested first reading sequence

1. [OpenVLA](https://proceedings.mlr.press/v270/kim25c.html): practical foundation and fine-tuning vocabulary.
2. [π0.5](https://proceedings.mlr.press/v305/black25a.html): semantic subtask supervision and broader real-world environments.
3. [Human-video survey](https://ijcai.org/proceedings/2026/871): how video information becomes robot-relevant.
4. [UniVLA: Learning to Act Anywhere](https://arxiv.org/abs/2505.06111) and [VIPA-VLA](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_Spatial-Aware_VLA_Pretraining_through_Visual-Physical_Alignment_from_Human_Videos_CVPR_2026_paper.html): latent actions and 3D alignment.
5. [VLM2VLA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7a0f8055c838df8e62329a76c7c6403d-Abstract-Conference.html): whether robot fine-tuning preserves semantic reasoning.
6. [ORION](https://ut-austin-rpl.github.io/ORION-release/) and [CoT-VLA](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.html): explicit relations versus predicted visual goals.

See the [paper collection](vla-papers.md) for recent additions, source links, evidence caveats and reading priorities. The [training roadmap](vla-training.md) describes what a first implementation would actually require.
