# GUIDELINES — Engineering Workflows

**Version:** 1.0.0  
**Status:** Normative  
**Scope:** All projects for your projects

---

## 1. Purpose

This document defines workflows for feature development, code review, and release processes. For technical standards and tool requirements, see [STANDARDS.md](STANDARDS.md).

---

## 2. Development Workflow

### 2.1 Issue-First Development

All work SHALL begin with an issue:

```bash
gh issue create --title "<description>" --body "<details>"
```

### 2.2 Branch Strategy

```
main ─────────●───────────●─────────────
               ↑           ↑
              PR₁         PR₂
               ↑           ↑
         cherry-pick   cherry-pick
               ↑           ↑
unstable/feat ──●──●──●──●──●──●──●──●──
```

| Branch Type | Purpose | Merges To |
|-------------|---------|-----------|
| `main` | Production-ready | — |
| `feature/*` | Clean PR history | `main` |
| `unstable/*` | Iteration workspace | `feature/*` via cherry-pick |

### 2.3 Atomic Commits & PRs

Commits and PRs SHALL be atomic:
- One logical change per commit
- One feature/fix per PR
- If PR grows too large, split into multiple PRs

**Never force push.** Instead:
```bash
# Create new branch from clean state
git checkout main && git pull
git checkout -b feature/issue-42-v2
git cherry-pick <good-commits>
# Close old PR, open new one
```

### 2.4 Feature Development Sequence

```bash
# 1. Create branches
git checkout main && git pull
git checkout -b feature/issue-42-user-auth
git checkout -b unstable/issue-42-user-auth

# 2. Iterate on unstable
# ... commit freely ...

# 3. Cherry-pick clean commits to feature
git checkout feature/issue-42-user-auth
git cherry-pick <hash>

# 4. Open PR
git push -u origin feature/issue-42-user-auth
gh pr create --fill

# 5. Continue on unstable while PR reviews
git checkout unstable/issue-42-user-auth
```

### 2.5 Bug Fix Sequence

```bash
# 1. Branch from main
git checkout -b bugfix/issue-43-null-check

# 2. Write failing test first
# 3. Implement fix
# 4. Verify

make prepush

# 5. Open PR
gh pr create --fill
```

---

## 3. Pre-Push Validation

Before every push, run:

```bash
make prepush
```

This target SHALL execute:
- Linting and formatting checks
- Type checking
- Tests

---

## 4. Code Review

### 4.1 Author Responsibilities

| Requirement |
|-------------|
| Self-review diff before requesting review |
| Ensure CI passes |
| Link issue in PR description |
| Respond to all comments |
| Do not force-push during active review |

### 4.2 Reviewer Responsibilities

| Requirement |
|-------------|
| Use Conventional Comments (see STANDARDS §4.4) |
| Approve when acceptable, not perfect |
| Trust author to address feedback |

---

## 5. Dependency Management

### 5.1 Adding Dependencies

```bash
# 1. Verify necessity (prefer stdlib/standard libraries)
# 2. Add dependency using your package manager
# 3. Commit lockfile
git add <lockfile> <manifest>
git commit -m "chore: add <package> for <purpose>"
```

---

## 6. Repository Setup

### 6.1 New Repository from Template

```bash
gh repo create <org>/<name> \
  --template <owner>/skills \
  --private \
  --clone

cd <name>
make install
pre-commit install
```

### 6.2 Post-Setup Customization

| File | Action |
|------|--------|
| `pyproject.toml` | Update name, description |
| `README.md` | Update project description |
| `PROJECT.md` | Document project context |

---

## 7. Release Process

```bash
# 1. Verify main is green
# 2. Update version in pyproject.toml
# 3. Update CHANGELOG.md
# 4. Tag and push
git tag v<major>.<minor>.<patch>
git push --tags
```

---

## 8. Context-Specific Guidelines

### 8.1 By Project Type

| Project Type | Guidelines |
|--------------|------------|
| Library/Package | Full workflow |
| Application | Full workflow |
| ML/Research | Full workflow; notebooks acceptable for exploration |
| Script/Prototype | Simplified; `make prepush` optional |

### 8.2 By Context

| Context | Standards |
|---------|-----------|
| Production | Full standards |
| Personal/Experimental | Lighter process acceptable |

---

## 9. Documentation

### 9.1 Token Budget

All documentation files SHALL be ≤8,000 tokens (~32KB, ~6,000 words).

| Reason | Benefit |
|--------|---------|
| LLM context limits | Docs fit in prompts |
| Cognitive load | Readable in one sitting |
| Forces concision | No rambling |

### 9.2 Structure

Each project SHOULD have:

| Document | Purpose | Target Size | Committed |
|----------|---------|-------------|-----------|
| README.md | Overview + links | 1-2k tokens | Yes |
| CONTRIBUTING.md | Contributor guide (humans + AI) | 500-1k tokens | Yes |
| GUIDELINES.md | Workflows (what + why) | 3-6k tokens | Yes |
| STANDARDS.md | Requirements (what + why) | 3-6k tokens | Yes |
| CLAUDE.md | Project-specific AI config | 500-1k tokens | Optional (unstaged) |

### 9.3 Worktree Standards

The working tree MAY include unstaged files for local configuration:

| File | Purpose | Staged |
|------|---------|--------|
| `CLAUDE.md` | Personal AI agent preferences | No |
| `PROJECT.md` | Project-specific context | Optional |
| `.env` | Environment variables | No |

Add to `.gitignore`:
```
CLAUDE.md
.env
```

AI agents SHOULD read `CLAUDE.md` if present, even when unstaged.

### 9.4 What + Why Pattern

GUIDELINES and STANDARDS sections SHOULD include both:

```markdown
## [Topic]

### What
Concrete rules, formats, examples.

### Why
Rationale, benefits, trade-offs.
```

### 9.5 Single Source

Content SHALL appear once, with pointers elsewhere:

```markdown
# ❌ Duplicate content
CONTRIBUTING: "Use conventional commits..."
GUIDELINES: "Use conventional commits..."

# ✅ Single source + pointer
GUIDELINES: "Use conventional commits..."
CONTRIBUTING: "See commit conventions in GUIDELINES.md"
```

---

## 10. References

- [STANDARDS.md](STANDARDS.md) — Technical requirements
- [CONTRIBUTING.md](CONTRIBUTING.md) — External contributor quick reference
