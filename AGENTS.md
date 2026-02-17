# AGENTS.md

AI agents: follow these repo rules. Keep changes minimal and scoped.

## Repo map
.
├─ AGENTS.md
├─ README.md
├─ docs/
├─ src/
│  ├─ expert_data_generation/
│  ├─ robot_kinematics/
│  └─ virtual_robot_environment/
└─ tests/

Quick notes:
- `README.md`: High level project plan.
- `docs/`: product + architecture documentation (start here for context).
- `src/expert_data_generation/`: All gode for expert data generation goes here.
- `src/robot_kinematics/`: All code related to the robot kinematics
- `src/virtual_robot_environment/`: Virtual environment used to train Policy (Do not change unless specificaly requested)
- `tests/`: automated tests live here.

## Guardrails
- Don’t refactor unrelated code.
- Don’t change deps, build, CI, or infra unless asked.
- Don’t touch secrets; never commit credentials.
- Ask/flag if requirements are ambiguous or risky.

## Output expectations
- Prefer clear diffs + brief rationale.
- Note files changed and commands run.
- Leave TODOs only when unavoidable and clearly scoped.
- Always comment code to ensure readability by humans.

Use a dedicated Git feature branch for each feature.
