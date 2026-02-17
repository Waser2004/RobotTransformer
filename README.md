# Robot Transformer Policy: Updated Training Plan

## Goal
Train a **Transformer-based policy** that maps a short **history of robot + environment state** to the **next action chunk**, enabling:
1. **Search** the workplate with an eye-in-hand camera and locate a specific cube  
2. **Grasp** the cube  
3. **Return + place** it at a target pose  
4. Maintain **safety constraints**: no self-collision, no robot–table/workplate collisions, no high-impulse contacts; **gripper–cube contact is allowed** during grasp

---

## 0) Control Interface (make the MDP well-defined)
### Time discretization
- Policy operates at a decision rate of `dt`, but outputs **action chunks**:
  - **Action chunk horizon:** `H` ticks
- `dt` = 20Hz
- `H`  = 20

### Action parameterization (recommended)
- Output **joint velocity targets** for the next `H` ticks:
  - `a_t = [v_{t:t+H-1}^1, ..., v_{t:t+H-1}^n, gripper_cmd_{t:t+H-1}]`
- Enforce actuator limits:
  - velocity limits, accel/jerk smoothing, gripper constraints

### Safety layer
Before executing any action chunk perform collision check.

---

## 1) Observations & Tokenization
### Per-timestep token content
At timestep `t`, build an observation token:
- `z_img_t`: visual embedding from the gripper camera image
- `q_t, qdot_t`: joint positions + joint velocities
- `g_t`: gripper state (open/close, width, force proxy if available)
- `a_{t-1}`: previous executed action (helps with partial observability / motion blur)
- `goal`: explicit target pose (or fixed goal token)

### History window
- Feed the Transformer a fixed history of `K` steps:
  - `[token_{t-K+1}, ..., token_t]`
- Add relative time encoding.

> Notes:
> - Eye-in-hand search is partially observable → history is mandatory.
> - If depth/segmentation is available in sim, use it for training signals and optionally as an auxiliary head.

---

## 2) Expert Data Generation (Supervised Pretraining / BC)
### Expert sources (better than purely scripted trajectories)
Use an expert that is robust and can generate recoveries:
- Motion planning (RRT*/PRM) for collision-free moves where appropriate
- IK/MPC for smooth local control
- Visual servoing for approach/alignment
- Systematic scan/search strategy or information gain heuristic

### Critical: recovery + disturbance coverage
Generate not only clean successes, but also:
- cube partially out of view / occluded
- camera noise, blur, lighting changes
- grasp failures (miss, slip), re-approach, re-center, re-scan
- slight perturbations to cube pose during rollout

### Domain + dynamics randomization (for sim robustness)
- Visual: textures, lighting, exposure, motion blur, sensor noise
- Geometry: cube pose/orientation, distractors, clutter, occlusions
- Dynamics: friction, restitution, cube mass, joint damping, latency

### Dataset structure
Store trajectories as sequences:
- `(o_{t-K+1:t}, a_{t:t+H-1}, info_t)`
Include labels for:
- phase/mode (optional): `search / approach / grasp / place / recover`
- safety margins (distance-to-collision) if available (for auxiliary learning)

---

## 3) Supervised Pretraining (Behavior Cloning)
### Model
- Transformer over token sequences
- Visual encoder (CNN or ViT) → `z_img_t`
- Policy head outputs **probabilistic continuous actions**:
  - per joint: mean + std (Gaussian) for velocities (or deltas)
  - train with negative log-likelihood (NLL)

### Losses
Primary:
- Action imitation loss over the chunk: `L_BC = NLL(a_expert | o_history)`

Auxiliary (recommended):
- Predict cube pose + confidence (if sim provides GT)
- Predict distance-to-collision margin (safety shaping)
- Optional phase classification head (if phases labeled)

### Evaluation protocol (pre-RL)
- Success rate across randomized conditions
- Recovery success under disturbances
- Collision rate / min clearance margins
- Time-to-success and path efficiency (baseline)

---

## 4) On-Policy Data Expansion (DAgger-style)
To reduce compounding errors and distribution shift:
1. Roll out the current policy in sim under randomization + perturbations
2. Query expert corrections for visited states
3. Aggregate dataset and continue training

Stop when:
- policy is robust to typical off-distribution states
- recovery behaviors are reliable

---

## 5) RL Improvement Stage (Efficiency + Robustness)
### Preferred ordering (stability)
1. **Offline RL** on the aggregated dataset to improve beyond expert and smooth behaviors
2. **Light online fine-tuning** (optional) to optimize speed/efficiency under constraints

### Reward design (avoid “just collision penalties”)
Components:
- **Success bonus**: cube placed within tolerance at target
- **Dense shaping**:
  - end-effector ↔ cube distance
  - cube ↔ target distance
  - alignment/orientation shaping for grasp/place
  - lift-height after grasp as a grasp-quality proxy
- **Efficiency**:
  - time penalty per step
  - energy/effort penalty (|action|, |accel|, jerk proxy)
- **Safety**:
  - penalize low clearance (distance-to-collision margins)
  - penalize disallowed contacts and high impulses

Collision rules:
- Allowed: gripper–cube contact during grasp
- Disallowed: robot–table/workplate, robot–self, high-impulse cube–table impacts, etc.

### Safety during RL
- Keep the safety shield enabled
- Curriculum tightening:
  - start with generous clearance, then tighten
  - start with easier cube placements, then expand distribution

---

## 6) Final Validation Checklist
- Randomized evaluation suite:
  - lighting/texture/dynamics variations
  - occlusions and distractors
  - forced perturbations mid-episode (small cube pushes, partial slips)
- Metrics:
  - success rate (overall + per-phase)
  - collision rate + min clearance
  - average time-to-success
  - smoothness (jerk proxy) and energy

---

## Deliverables / Milestones
1. **MDP finalized:** dt, H, action limits, collision classes
2. **Expert generator:** planning + visual servo + recovery scripts
3. **BC baseline:** robust success under randomization
4. **DAgger loop:** distribution shift mitigated
5. **RL improvement:** faster, smoother, same-or-better safety
6. **Sim2real readiness:** strong randomization, safety shield, calibration robustness