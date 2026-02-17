# Expert Data Generation

## Generating Optimal examples
1. **Initialize “cube-at-home, robot-at-home” start states (search trigger examples)**
    - Call `reset(cube_position="home", robot_pose="home", actuator_rotations=None)`.
    - Capture several observation frames of the cube in its home position (including slight viewpoint variation if available).
    - Purpose: provide examples where the model learns that *if the cube is not visible from the home pose*, it should initiate the search routine.
2. **Initialize randomized task start states**
    - Call `reset(cube_position="random_on_workplate", robot_pose="home", actuator_rotations=None)`.
3. **Search phase (checkpoint sweep)**
    - Execute the search trajectory defined in `search_path.json` (a sequence of poses that serve as waypoints).
    - After each checkpoint, evaluate visibility:
        - If `target_cube_in_view(padding=0.1)` returns `true`, terminate the search phase and transition to **Approach**.
        - Otherwise, continue to the next checkpoint.
4. **Approach phase (keep target in view)**
    - Calculate pre-grasp position using inverse kinematics and the following inputs:
        
        ```python
        # location
        self.location = [
          x_pos, 
          y_pos, 
          25
        ]
                
        # rotation
        distance_to_target = math.sqrt(self.location[0] ** 2 + self.location[1] ** 2)
        self.rotation = [
          -150 if distance_to_target > 440 else -90, 
          0, 
          90 + z_rot
        ]
        ```
        
    - From the current pose, move to a pre-grasp pose that aligns the gripper for a stable grasp.
    - Constraint: maintain visual contact with the cube throughout the approach (when feasible) to reduce the chance of losing the target due to occlusion or incidental cube motion.
5. **Grasp and return phase**
    - When the pre-grasp pose is reached, close the gripper to grasp the cube.
    - Transport the cube to the home position, release it, then return the robot to `robot_pose="home"`.

## Generating Failure Cases for Robustness
Failure data is intentional training data, not noise. The purpose is to teach the Transformer robust recovery behavior under realistic breakdowns, not only to record that failures happened.

### Failure Design Principles
1. **Phase-aware failure coverage**
    - Generate failures explicitly in `search`, `approach`, `grasp`, `return/place`, and `recover`.
2. **One primary disturbance per episode**
    - Keep one dominant failure driver per rollout to avoid confounded labels and unclear causality.
3. **Recovery-first expert behavior**
    - Expert logic should attempt bounded repair actions (re-scan, re-approach, re-grasp) before declaring failure.
4. **Safety constraints remain active**
    - Unsafe collisions are never treated as valid successful behavior and should trigger abort handling.
5. **Balanced modality coverage**
    - Maintain balanced counts across visual failures, kinematic/pose failures, and manipulation/contact failures.

### Phase-by-Phase Failure Matrix
| Phase | Failure Case | How to Induce in Sim | Expected Expert Recovery | Episode Outcome Label |
|---|---|---|---|---|
| search | Cube out of FOV at home | Start from `reset(cube_position="random_on_workplate", robot_pose="home")` with cube outside initial camera frustum. | Execute `search_path.json` checkpoints until `target_cube_in_view(padding=0.1)` becomes true, then transition to approach. | `success_direct` or `success_after_recovery` |
| search | Near-edge visibility flicker | Place cube near image boundary so visibility toggles with small pose changes. | Require stable re-detection over multiple checkpoints before committing to approach. | `success_after_recovery` |
| search | Lighting glare / low contrast | Use extreme randomized light intensity and color combinations from environment reset. | Continue scanning with conservative visibility confirmation before phase switch. | `success_after_recovery` or `failure_recoverable_not_resolved` |
| search | Distractor-like clutter confusion | Introduce non-target cube-like distractors in view while target is partially visible/occluded. | Keep search sweep and only commit when target identity confidence is stable across views. | `success_after_recovery` or `failure_unrecoverable` |
| search | Temporary occlusion | Briefly occlude camera line-of-sight during checkpoint sweep. | Hold or continue to next checkpoint, then reacquire before approach. | `success_after_recovery` |
| approach | Intermittent target loss during move | Inject short occlusion or visibility dropout while moving to pre-grasp. | Pause forward progress, perform micro re-centering and resume approach after re-lock. | `success_after_recovery` |
| approach | Cube drift while approaching | Apply small cube position nudge mid-approach. | Recompute pre-grasp target and execute corrected approach trajectory. | `success_after_recovery` |
| approach | Near-workspace-limit pre-grasp | Sample cube near reachable boundary and high-arm-extension region. | Attempt safe alternate approach orientation; abort if safety/IK feasibility is violated. | `success_after_recovery` or `failure_unrecoverable` |
| approach | High relative rotation misalignment | Set cube yaw requiring difficult wrist alignment for stable grasp. | Perform alignment correction before descent and grasp attempt. | `success_after_recovery` or `failure_recoverable_not_resolved` |
| grasp | Lateral gripper misalignment | Offset final approach laterally by a small controlled bias. | Re-open, back off, and perform a corrected re-approach + re-grasp. | `success_after_recovery` |
| grasp | Yaw misalignment at contact | Rotate cube yaw near grasp tolerance limits. | Rotate/reposition end effector and retry grasp with corrected orientation. | `success_after_recovery` or `failure_recoverable_not_resolved` |
| grasp | Partial finger contact | Set descent depth/contact timing so only one finger gets stable contact initially. | Reopen and retry with corrected centering and closure timing. | `success_after_recovery` |
| grasp | Slip on initial lift | Reduce effective grip quality/friction during first lift movement. | Detect slip, lower safely, re-acquire stable grasp, and retry lift. | `success_after_recovery` or `failure_unrecoverable` |
| grasp | Delayed/weak close timing | Delay or weaken close command around grasp window. | Execute bounded re-grasp attempts with corrected close timing. | `success_after_recovery` or `failure_recoverable_not_resolved` |
| return/place | Post-grasp slip while carrying | Inject carry disturbance after successful lift. | Transition to recovery: track dropped cube, re-scan, and re-enter approach/grasp. | `success_after_recovery` or `failure_unrecoverable` |
| return/place | Drop-and-reacquire | Force release/drop event before reaching home/target zone. | Trigger recovery loop with target reacquisition and repeat transport. | `success_after_recovery` |
| return/place | Target-zone overshoot | Add controlled pose overshoot near place location. | Correct with bounded backtracking and re-place attempt. | `success_after_recovery` |
| return/place | Unstable carry trajectory | Induce trajectory wobble that risks object loss. | Slow down trajectory and prioritize stable transport over speed. | `success_after_recovery` or `failure_safety_abort` |
| recover | First grasp failure then re-approach | Force first grasp attempt miss. | Execute explicit recover sequence: back off, re-center, retry once/twice. | `success_after_recovery` or `failure_recoverable_not_resolved` |
| recover | Lost visual lock then re-scan | Remove target from view after approach begins. | Re-enter search checkpoints until stable detection and then resume task. | `success_after_recovery` |
| recover | Fallback to search checkpoints | Trigger repeated local recovery failure in approach/grasp. | Escalate to full search routine instead of persisting unsafe local retries. | `success_after_recovery` or `failure_unrecoverable` |

### Disturbance Injection Strategy
1. **Pre-episode randomization**
    - Randomize lighting and target cube pose before rollout start.
2. **Mid-episode perturbations**
    - Inject small cube nudges or brief occlusions during search and approach.
3. **Post-grasp perturbations**
    - Stress carry robustness with slip/drop-oriented disturbances after successful grasp.
4. **Severity bands**
    - Mild: default training distribution; mostly recoverable with one correction.
    - Moderate: regular robustness training; requires deliberate recover behavior.
    - Hard: curriculum tail; may require multiple recover transitions or safe abort.
5. **Hard-case cap**
    - Hard disturbances must be capped so dataset quality remains learnable and not dominated by unrecoverable chaos.

### Sampling and Curriculum
1. **Target sampling mix**
    - 50% nominal successful trajectories.
    - 35% recoverable failures requiring corrective behavior.
    - 15% hard failures or aborts (safety-terminated or unrecoverable).
2. **Curriculum order**
    - Stage 1: nominal + mild disturbances.
    - Stage 2: recoverable moderate failures with recovery-first expert behavior.
    - Stage 3: hard edge cases under strict safety filtering and bounded retries.

### Outcome Labels and Episode End States
1. `success_direct`
    - Task completed without requiring an explicit recovery transition.
2. `success_after_recovery`
    - Task completed after one or more bounded recovery behaviors (for example re-scan, re-approach, or re-grasp).
3. `failure_recoverable_not_resolved`
    - Failure mode was theoretically recoverable, but recovery budget was exhausted before success.
4. `failure_unrecoverable`
    - Episode entered a condition that cannot be solved within defined constraints (for example unreachable/invalid pose progression).
5. `failure_safety_abort`
    - Episode terminated because safety constraints required abort (collision risk, unsafe contact class, or invalid safety margin).

### Robustness Acceptance Criteria
1. Recovery success rate must improve over dataset-generation iterations.
2. Collision and safety-abort rate must remain below a strict, pre-defined ceiling.
3. Search-to-first-detection latency must be tracked and remain within bounded targets.
4. Grasp stability failures (slip/drop) must be tracked separately from detection/localization failures.
5. All metrics must be reported per phase and per failure type, not only as aggregate totals.

### Data Quality Gates
1. Exclude trajectories showing impossible geometry or simulation glitches.
2. Exclude trajectories with invalid state transitions or phase order violations.
3. Exclude trajectories with ambiguous or contradictory outcome labeling.
4. Perform periodic sampled review with this checklist:
    - Visual plausibility of target/camera interaction.
    - Recovery behavior realism (not random oscillation).
    - Safety compliance throughout the episode.
