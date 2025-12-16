# Project Continuation Prompt

Copy everything below the line and paste it into a new conversation to resume work on this project.

---

## PROMPT START (copy from here)

I am building an end-to-end movie recommendation system as a learning project. I have detailed documentation that defines the complete project. Please read all attached documents carefully before responding.

### Project Documents

I will attach the following files to this message:
1. **PROBLEM_STATEMENT.md** — Defines objectives, requirements, constraints, success metrics
2. **ARCHITECTURE.md** — System design, component details, data flows, API spec, directory structure
3. **IMPLEMENTATION_PLAN.md** — Detailed phase/step breakdown with completion status

### Current Status

<!-- UPDATE THIS SECTION EACH TIME YOU START A NEW CONVERSATION -->

**Currently on:** Phase 1, Step 1.7

**Completed steps:**
- Phase 1: Steps 1.1 through 1.6 ✅

**Files already created:**
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- Directory structure with all `__init__.py` files
- `.gitignore`
- `configs/data.yaml`
- `configs/training.yaml`
- `configs/serving.yaml`
- `configs/monitoring.yaml`
- `configs/retrain.yaml`
- `Makefile`

**Next step to implement:** Step 1.7 — Create `.pre-commit-config.yaml`

### How I Want to Work

1. **Step-by-step approach:** Before writing code for any step, first explain what we're building and why it matters for the overall system.

2. **One step at a time:** Complete and verify each step before moving to the next.

3. **Teaching style:** Explain concepts as if I'm learning how to build production ML systems. Include:
   - What the component does
   - Why it's needed
   - How it connects to other parts
   - Key decisions and trade-offs

4. **Verification:** After each step, provide commands to verify the implementation works.

### Technical Setup

- **OS:** macOS
- **Python:** 3.9.6
- **Package manager:** pip + conda (for packages with C dependencies like `implicit`)
- **Virtual environment:** Using `.venv`

### Request

Please confirm you've understood the project by briefly summarizing:
1. What we're building
2. Current phase and step
3. What we'll implement next

Then proceed with the next step using the teaching approach described above.

## PROMPT END

---

## How to Use This Prompt

### Before starting a new conversation:

1. **Update the "Current Status" section** with:
   - Which phase/step you're on
   - List of completed steps
   - List of files already created
   - What step comes next

2. **Attach all three documents** to your message:
   - `PROBLEM_STATEMENT.md`
   - `ARCHITECTURE.md`
   - `IMPLEMENTATION_PLAN.md`

3. **Copy the prompt** (everything between PROMPT START and PROMPT END)

4. **Paste into new conversation** along with the attached files

### Example: Resuming at Phase 2, Step 2.3

```markdown
**Currently on:** Phase 2, Step 2.3

**Completed steps:**
- Phase 1: All steps ✅
- Phase 2: Steps 2.1, 2.2 ✅

**Files already created:**
- All Phase 1 files
- `src/config.py`
- `src/data/download.py`

**Next step to implement:** Step 2.3 — Implement preprocessing
```

### Example: Resuming at Phase 4, Step 4.1

```markdown
**Currently on:** Phase 4, Step 4.1

**Completed steps:**
- Phase 1: All steps ✅
- Phase 2: All steps ✅
- Phase 3: All steps ✅

**Files already created:**
- All Phase 1, 2, 3 files
- Trained model in `models/production/`

**Next step to implement:** Step 4.1 — Define Pydantic schemas
```

---

## Tips for Effective Continuation

1. **Keep IMPLEMENTATION_PLAN.md updated** — Mark steps as ✅ when complete

2. **Note any deviations** — If you made changes to the architecture or added files not in the plan, mention them in the prompt

3. **Include error context** — If you're resuming because of an error, include the error message in your prompt

4. **Attach relevant code files** — If debugging or modifying existing code, attach those specific files

5. **Be specific about what you need** — "Continue from step X" is clearer than "continue the project"
