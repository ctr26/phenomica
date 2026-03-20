# CONTRIBUTING — Contributor Guide

> For humans and AI agents. Full specification: [GUIDELINES.md](GUIDELINES.md).

## For AI Agents

**Always use skills before acting.** Run through core skills:
- `what` — understand requirements
- `why` — understand rationale  
- `where` — find the right location
- `how` — find prior art, plan approach

**Always contextualize via root docs.** Before non-trivial work:
1. Read README.md → Contents → relevant docs
2. Cite STANDARDS.md or GUIDELINES.md to justify decisions
3. If no doc supports the decision, flag as new convention

**Policy changes SHALL be versioned.** When modifying rules/standards:
1. Increment version in frontmatter
2. Note change in commit message
3. Consider backward compatibility

**Standardise iteratively:**
- Small patches, not big rewrites
- One section at a time
- Commit after each change
- Information SHALL NOT be lost
- Minimize token/char count (tables > prose)

Before starting work, ask clarifying questions:

| Question | Skill | Purpose |
|----------|-------|---------|
| **What** are the rules? | `what/` | Understand requirements |
| **Why** do we do this? | `why/` | Understand rationale |
| **Where** does this apply? | `where/` | Understand context/exceptions |
| **How** do I do this? | `how/` | Understand workflow |
| **Who** should review? | `who/` | Find the right reviewer |

## Project-Specific Agent Config

Create `CLAUDE.md` (or `AGENTS.md`) for project-specific AI instructions. This file is **optional and typically unstaged** — add to `.gitignore`.

### Hierarchical Config

AGENTS.md SHOULD reference parent configs:

```markdown
# AGENTS.md
> Parent: [~/AGENTS.md](~/AGENTS.md)

## Project Overrides
- Focus: this project's domain
```

Inheritance: `~/AGENTS.md` → domain → project

---

## Quick Start

```bash
gh repo fork <owner>/<repo> --clone
cd PROJECT
make install
git checkout -b feature/issue-N-description
# ... make changes ...
make prepush  # MUST pass
gh pr create --fill
```

## Commands

```bash
make install    # Setup environment
make prepush    # Before every push (REQUIRED)
make lint       # Type check + lint
make test       # Run tests
```

## Non-Negotiable

| Rule | Enforcement |
|------|-------------|
| Conventional Commits | `feat:`, `fix:`, `docs:`, `test:`, `chore:` |
| Type annotations | On exports and public APIs |
| Pre-push checks | `make prepush` must pass |
| Tests | Unit tests for core functionality |

## Workflow

1. Issue first
2. Branch: `feature/issue-N-desc`
3. `make prepush` must pass
4. PR links issue

## Checklist

- [ ] Issue exists and is linked
- [ ] Conventional commits used
- [ ] PR labeled (use `agent` for AI-generated PRs)
- [ ] Type hints complete
- [ ] Doctests added for public functions
- [ ] `make prepush` passes

## Commit Format

```
<type>: <description>

Types: feat | fix | docs | test | chore | refactor
```

## Review Comments

Use [Conventional Comments](https://conventionalcomments.org/):

| Prefix | Meaning |
|--------|---------|
| `issue:` | Must fix |
| `suggestion:` | Consider |
| `question:` | Clarify |
