# HUMANS.md

**If this file exists, this repository has been standardised.**

---

## What This Means

This repo follows a documentation system designed for both humans and AI agents.

### For Humans

You already know how to read docs. Nothing changes for you — README, CONTRIBUTING, GUIDELINES, and STANDARDS work exactly as you'd expect.

### For Agents

AI agents (Claude, Codex, Copilot, etc.) can use lightweight **skills** to navigate this repo:

| Skill | Purpose |
|-------|---------|
| `what` | Understand requirements and constraints |
| `why` | Understand rationale and design decisions |
| `where` | Find the right location for changes |
| `how` | Find workflows and processes |
| `who` | Find relevant reviewers |
| `standardise` | Set up or improve repo docs |

**Agents should automatically:**
- Read these docs before writing code
- Write code that complies with STANDARDS
- Follow GUIDELINES workflows
- Use conventional commits
- Request appropriate reviewers

No prompting needed. Compliance is automatic.

---

## Design Goals

| Goal | How |
|------|-----|
| **General** | Works for any language, any project |
| **Convenient** | Uses docs you already have (or provides templates) |
| **Pervasive** | Same system across all your repos |
| **Low lift** | Add once, benefit forever |

### Zero Configuration

Agents that understand this system will:
1. Detect HUMANS.md → know repo is standardised
2. Read root docs automatically
3. Write compliant code without being asked
4. Follow commit conventions
5. Create properly labeled PRs

You don't train agents per-repo. You standardise once.

### Optimised for Both Audiences

The `standardise` skill rebalances documentation:

| For Humans | For Agents |
|------------|------------|
| Info in docs humans actually read (README, CONTRIBUTING) | Each doc ≤8k tokens (fits context window) |
| Important stuff surfaces to top | Consistent structure for parsing |
| Less hunting through long files | Links instead of duplication |
| Single source of truth | No conflicting information |

**Deduplication:** Information lives in one place. Other docs link to it. No copy-paste drift, no contradictions.

**Result:** Humans find what they need. Agents can read entire docs.

---

## Philosophy

### Conventions First

The system leans on existing conventions:
- Conventional Commits
- README → CONTRIBUTING → GUIDELINES → STANDARDS hierarchy
- Issue-first development
- PR templates with context

Agents don't need special instructions — they read the same docs humans read.

### Lightweight Skills

Skills are thin wrappers, not heavy rulesets. They teach agents:
1. Where to look (root dir docs)
2. How to interpret what they find (SHOULD vs SHALL)
3. When to ask for help (missing docs → suggest `standardise`)

### Human-Agent Symmetry

| Human | Agent |
|-------|-------|
| Reads README | Reads README via `what` skill |
| Follows CONTRIBUTING | Follows CONTRIBUTING via `how` skill |
| Asks reviewer | Uses `who` skill to find reviewer |
| Checks STANDARDS | Checks STANDARDS via `what` skill |

Same docs. Different readers.

---

## The System

```
┌─────────────────────────────────────────┐
│              Root Docs                  │
│  README → CONTRIBUTING → GUIDELINES     │
│              → STANDARDS                │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼────┐        ┌─────▼─────┐
   │  Human  │        │   Agent   │
   │ (reads) │        │  (skill)  │
   └─────────┘        └───────────┘
```

One source of truth. Two audiences.

---

## How to Standardise a Repo

### Context Detection

```bash
# Check if in git repo
git rev-parse --git-dir 2>/dev/null
```

| Context | Doc Location | Structure |
|---------|--------------|-----------|
| Git repo | Root dir (`/`) | README → CONTRIBUTING → GUIDELINES → STANDARDS |
| Non-git | Near skills | SKILL.md can embed docs or use local README.md |

### Core Principle

**DO NOT modify existing markdown files.**

Standardise is additive only:
- Copy templates for MISSING files
- Leave existing docs untouched
- Work with what the repo already has

### Step 1: Init Missing Docs

```bash
# Check what exists
ls -la *.md 2>/dev/null

# Only copy missing docs from templates
for doc in README.md CONTRIBUTING.md GUIDELINES.md STANDARDS.md HUMANS.md; do
  [ ! -f "$doc" ] && cp ./docs/"$doc" . && echo "Created $doc"
done
```

**Copy from templates ONLY if missing:**

| File | Copy if missing |
|------|-----------------|
| README.md | Yes |
| CONTRIBUTING.md | Yes |
| GUIDELINES.md | Yes |
| STANDARDS.md | Yes |
| HUMANS.md | Yes |

**Never overwrite existing docs.** Existing docs represent the repo owner's decisions.

### Step 2: Use Existing Structure

If docs already exist, work with them:
- Read existing README for project context
- Follow existing CONTRIBUTING workflow
- Respect existing GUIDELINES and STANDARDS

**Skills SHALL link to README only.** README is the search entry point:

```
Agent query → Skill → README → Contents → Target doc
```

### Step 3: Suggest Improvements (Don't Enforce)

If existing docs could be improved, **suggest** changes to the user:

| Observation | Suggestion |
|-------------|------------|
| Doc >8k tokens | "Consider splitting this doc" |
| Missing README | "No README found — copy template?" |
| No CONTRIBUTING | "No contributor guide — copy template?" |

**Do not modify existing content.** Report findings and let the user decide.

```bash
# Check token counts (~0.75 words per token)
wc -w *.md | awk '{print $2, int($1/0.75), "tokens"}'
```

### Step 4: Hierarchical AGENTS.md

AGENTS.md (unstaged) SHOULD reference parent configs:

```markdown
# AGENTS.md

> Parent: [~/AGENTS.md](~/AGENTS.md)

## Project Overrides

- Focus: ML research
- Extra tools: dvc, wandb
```

Inheritance: `~/AGENTS.md` → `~/domain/AGENTS.md` → `~/domain/project/AGENTS.md`

### Step 5: Resolve Contradictions

Before finalizing, check for conflicts:

| Check | Action |
|-------|--------|
| GUIDELINES vs STANDARDS | Align or document exception |
| README claims vs actual | Update README |
| Old content vs new | Remove stale, keep current |
| Skill vs root docs | Root docs are authoritative |

Flag unresolved contradictions with:
```markdown
> ⚠️ **Contradiction:** [description] — needs resolution
```

### Verification Checklist

- [ ] README links all docs
- [ ] Each doc ≤8k tokens
- [ ] No duplicate content
- [ ] No contradictions (or flagged)
- [ ] GUIDELINES has what+why
- [ ] STANDARDS has what+why
- [ ] PR labeled (`agent` for AI PRs)
- [ ] `make prepush` passes

---

## If You're New Here

1. Start with **README.md** — project overview
2. Read **CONTRIBUTING.md** — how to contribute
3. Check **GUIDELINES.md** — workflows and processes
4. Reference **STANDARDS.md** — technical requirements

If you're an agent, use the skills. If you're human, just read the docs.

---

*Standardised by [THE SYSTEM](https://github.com/ctr26/skills)*
