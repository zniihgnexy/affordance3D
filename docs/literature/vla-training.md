# A practical roadmap for training and using a VLA

This is a starting research plan, not a robot-specific installation guide. The best model, action representation and compute budget depend on the robot, cameras, tasks, data and available hardware.

## How training generally works

1. **Start from a pretrained vision-language or VLA checkpoint.** It supplies prior visual and language representations and, for a VLA, may already include robot behavior.
2. **Prepare demonstrations.** Each episode generally needs time-aligned camera frames, robot state, actions, and a task instruction. Track action units, coordinate frames, gripper convention, camera names, sampling frequency and episode boundaries. For new robots, follow the chosen framework's exact schema.
3. **Choose an adaptation method.** Fine-tune the whole policy, train selected components, or use parameter-efficient adapters such as LoRA where supported. Some models output discrete action tokens; others predict continuous action chunks. The required labels and inference code depend on the checkpoint.
4. **Evaluate offline and in a closed loop.** Use held-out episodes to diagnose data and prediction errors, then run the policy with new observations feeding back into subsequent decisions. Offline similarity alone does not establish task success.
5. **Deploy with supervision.** Connect model observations and actions to the robot's actual sensors and controller; confirm units, coordinate frames, limits, timing and safe stopping in the target setup. A pretrained checkpoint is not automatically compatible with an arbitrary embodiment.

Videos without robot actions can provide valuable pretraining signals, but cannot be treated as if they directly contain the target robot's joint commands. Methods such as latent actions, tracking, 3D reconstruction or robotized demonstrations introduce additional assumptions and still need an embodiment-grounding path. See the [human-video survey](https://ijcai.org/proceedings/2026/871).

## Practical places to start

| Option | Best use | What to check |
| --- | --- | --- |
| [LeRobot + SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla) | Accessible end-to-end demonstration collection, fine-tuning and deployment on supported setups. | Supported robot, schema, camera setup and compute; the guide's example demonstration count and training time are examples, not a guarantee. |
| [OpenVLA](https://github.com/openvla/openvla) + [OpenVLA-OFT](https://openvla-oft.github.io/) | Research baseline with published training and inference tools; useful for controlled comparisons. | GPU memory, robot-specific preprocessing, action normalization and checkpoint compatibility. OFT is a fine-tuning recipe. |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | Experiment with available π-family checkpoints and training examples. | Dataset conversion and embodiment; the repository's examples are not a complete reproduction of the original π0.5 system and data. |
| [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) | Investigate broader embodiment and humanoid-oriented adaptation. | Specific model version, license, GPU needs, supported sensors/actions and deployment interface. |

Check each repository's current instructions, checkpoint license, dependencies and actual hardware requirements before committing to a choice. No code has been installed or trained for this collection.

## A first experiment aligned with your idea

1. Pick a concrete robot and one manipulation task with measurable success, such as moving an object to a target location. Decide whether the robot is stationary or mobile and what it can observe.
2. Collect demonstrations across a range of object positions, camera views and backgrounds. Record task instructions and robot actions along with synchronized video. Keep some rooms, objects and instruction paraphrases entirely out of training.
3. Fine-tune an available baseline and evaluate it with identical settings across the held-out cases. Record number of trials, successes, failure categories, interventions and timing.
4. Add a specific hypothesis: for example, augment the same baseline with an object relationship or contact/clearance representation, or add human-video pretraining. Keep data and compute as comparable as possible.
5. Test whether gains persist across new scenes, new object instances, language paraphrases, distractors and longer task sequences. Measure both task success and separate semantic grounding questions; success in one does not prove the other.

For reliable conclusions, compare like with like: robot embodiment, observation cameras, action frequency, demonstrations, training budget and held-out tasks. Simulation and real-world results should be reported separately. Adapt the safety and stopping procedure to the actual hardware before physical testing.

## Open choices for the project

The next useful information is the desired robot and action domain, available cameras and compute, whether real hardware is available, and whether the first aim is to understand papers, reproduce an existing model, or test a new research idea. Those decisions determine which framework and datasets are appropriate.
