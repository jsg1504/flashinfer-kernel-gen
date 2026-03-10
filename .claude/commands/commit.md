# /commit Command

**Scope**: project

---

## Command Instructions

Create a git commit for the current kernel optimization changes.

## Steps

1. **Check status** — Run `git status` and `git diff --stat` to see what changed.

2. **Determine commit type** — Based on the changes:
   - Kernel optimization → `opt(gdn-decode): Ox — description`
   - Workflow/tooling → `chore: description`
   - Documentation only → `docs: description`
   - Bug fix → `fix(gdn-decode): description`

3. **Read optimize.md** — Check the latest attempt entry to extract:
   - Which optimization (Ox) was applied
   - Correctness results (N/20 PASSED)
   - Performance results (latency, speedup)

4. **Stage files** — Only stage relevant files:
   - `solution/triton/kernel.py` — kernel changes
   - `optimize.md` — optimization tracking
   - `know-how.md` — technique documentation
   - `.claude/` — command/skill changes (if any)
   - **DO NOT** stage `solution.json` (generated artifact, gitignored)
   - **DO NOT** stage `mlsys26-contest/` (read-only submodule)
   - **DO NOT** stage `.env` or credentials

5. **Commit message format** — Use this template:

   ```
   opt(gdn-decode): Ox — short description

   [1-3 lines explaining what changed and why]

   Changes:
   - [bullet list of specific changes]

   Correctness: N/20 PASSED, abs_err=X.XXe-XX
   Performance: [improvement/neutral/regression] on [GPU name]
   ```

6. **Create commit** — Run `git add <files> && git commit -m "..."`.

7. **Report** — Show the commit hash and summary.

## Rules

- NEVER force push or amend unless explicitly asked
- NEVER commit `solution.json`, `.env`, or submodule changes
- If no changes to commit, say so and stop
- Match the existing commit message style in the repo
