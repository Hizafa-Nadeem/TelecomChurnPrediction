# CLAUDE.md

## Project Goal

**Current milestone:** Build v1: build, train, evaluate, explain, package and serve the churn model entirely on the local machine.

## Project Docs

The docs are the single source of truth. Don't restate their content here.

- `docs/project_spec.md`: what to build. §5 Tech Stack, §6 Repository Structure, §4 Azure-Ready Design Rules, §9 Acceptance Criteria, phases (§3 v1 local, §7 v2 Azure demo, §11 v3 optional).
- `docs/development_strategy.md`: how to build it. Dataset properties, EDA plan and Steps 1-16.
- `docs/Azure_production_plan.md`: the Azure demo plan (v2). Not for v1.

## Coding Instructions

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Code Style

- Type hints and short docstrings on public functions.
- Comments only for the "why", not the "what".

## Environment

- Use the project virtual environment `.venv` and install from `requirements-dev.txt`.
- To add a package, edit `requirements.in` (or `requirements-dev.in`) and update the pins in the matching `.txt` file.
- Settings and secrets come from `.env` (template: `.env.example`). Model and pipeline parameters live in `configs/params.yaml`.
- NEVER commit `.env`, `data/` or `experiments/`.

## GitHub Instructions

### Repository Etiquette

**Branching:**
- Always create a feature branch before starting major changes
- NEVER commit directly to `main`
- Branch naming: `feature/description` or `fix/description`

**Git workflow for major changes:**
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Develop and commit on the feature branch
3. Test locally before pushing the code
4. Push the branch: `git push -u origin feature/your-feature-name`
5. Create a PR to merge into `main`
6. Update docs alongside code changes (`docs/project_status.md`) in the same commit

**Commits:**
- Write clear commit messages describing the change
- Keep commits focused on single changes

**Pull Requests:**
- NEVER force push to `main`
- Include description of what changed and why
