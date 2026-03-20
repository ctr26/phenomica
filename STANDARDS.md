# STANDARDS — Engineering Technical Standards

**Version:** 1.0.0  
**Status:** Normative  
**Scope:** Software projects

---

## 1. Definitions

| Term | Definition |
|------|------------|
| **SHALL** | Absolute requirement |
| **SHALL NOT** | Absolute prohibition |
| **SHOULD** | Recommended unless justified |
| **MAY** | Optional |

---

## 2. Quality Practices

### 2.1 Required Practices

Projects SHALL implement:

| Practice | Purpose |
|----------|---------|
| Type checking | Catch errors at analysis time |
| Linting | Consistent style, catch bugs |
| Unit tests | Verify individual components |
| Pre-commit checks | Catch issues before push |

### 2.2 Recommended Practices

Projects SHOULD implement:

| Practice | Purpose |
|----------|---------|
| Smoke tests | Verify basic functionality |
| Integration tests | Verify component interaction |
| Coverage tracking | Measure test completeness |

### 2.3 User Discretion

Full test coverage targets are at user discretion based on:
- Project criticality
- Team capacity
- Risk tolerance

### 2.4 Common Targets

Projects SHOULD implement consistent entry points:

```bash
make install    # or: npm install, uv sync, etc.
make lint       # Static analysis
make test       # Test execution  
make prepush    # Pre-push validation
```

Tool choice is flexible — use what fits your ecosystem.

---

## 3. Code Quality

### 3.1 Type Safety

| Requirement | Enforcement |
|-------------|-------------|
| Type annotations | Type checker of choice |
| Explicit types on exports | Code review |
| Avoid `any`/`Any` | Justify with comment if needed |

### 3.2 Testing

| Level | Purpose | Required |
|-------|---------|----------|
| Unit tests | Component correctness | Yes |
| Smoke tests | Basic functionality | Recommended |
| Integration tests | System interaction | Project-dependent |
| Coverage targets | Measure completeness | User discretion |

Flaky tests SHALL be fixed or deleted.

### 3.3 Architecture Constraints

| Constraint | Rationale |
|------------|-----------|
| Shallow inheritance | Easier to understand |
| Functions over classes | When stateless, prefer simplicity |
| Validate at boundaries | Fail fast, trust internal data |

### 3.4 Style

| Guideline | Rationale |
|-----------|-----------|
| Consistent line length | Readability |
| Docstring/comment convention | Project consistency |
| Import ordering | Automated via tooling |

---

## 4. Version Control

### 4.1 Commit Messages

Format: `<type>: <description>`

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `test` | Test addition/modification |
| `chore` | Maintenance |
| `refactor` | Code restructure |

Constraints:
- Present tense ("add" not "added")
- Lowercase
- No trailing period
- Reference issue when applicable: `feat: add auth (#42)`

### 4.2 Branch Naming

Format: `<type>/issue-<N>-<description>`

| Type | Usage |
|------|-------|
| `feature` | New functionality |
| `bugfix` | Defect correction |
| `chore` | Maintenance |
| `docs` | Documentation |
| `experimental` | Exploratory work |

### 4.3 Pull Requests

| Requirement | Enforcement |
|-------------|-------------|
| Issue linkage | PR template |
| CI passage | Branch protection |
| Description | PR template |
| Merge method | Squash and merge |

### 4.4 PR Labels

PRs SHALL use conventional labels:

| Label | Usage |
|-------|-------|
| `agent` | AI-generated PR (required for agent PRs) |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `chore` | Maintenance |
| `refactor` | Code restructure |
| `test` | Test changes |

Agent-driven PRs SHALL be tagged with `agent`:
```bash
gh pr create --label agent
```

Agent branches MAY use: `agent/<agent-name>/<description>`

### 4.5 Review Comments

Prefix format per [Conventional Comments](https://conventionalcomments.org/):

| Prefix | Meaning | Action Required |
|--------|---------|-----------------|
| `issue:` | Defect | Yes |
| `suggestion:` | Improvement | Optional |
| `question:` | Clarification needed | Response |
| `nitpick:` | Minor style | Optional |
| `praise:` | Positive feedback | None |

---

## 5. Security

| Requirement | Enforcement |
|-------------|-------------|
| No hardcoded credentials | `gitleaks` in CI |
| Secrets via environment | Review |
| Private repos by default | Repository settings |

---

## 6. Design Patterns

### 6.1 Boundary Validation

Validation SHALL occur at system boundaries (entry points, API handlers, config loading):

```
✅ Validate input at entry → trust internally
❌ Validate mid-execution → too late, error already propagated
```

Benefits:
- Fail fast with clear errors
- Internal code can trust data
- Single validation point

### 6.2 Schemas as Test Source

Strong schemas SHOULD serve as source of truth for test data:

| Benefit | Mechanism |
|---------|-----------|
| Single source of truth | Schema defines valid input space |
| No fixture drift | Factory uses schema validation |
| Self-documenting | Constraints visible in schema |

Anti-patterns:
- Separate test constants that drift from schema
- Hardcoded values that may become invalid

### 6.3 General Patterns

| Pattern | Rationale |
|---------|-----------|
| Function components | Simpler, more predictable |
| Named exports | Clearer imports |
| Strict type checking | Catch errors early |

---

## 7. Exceptions

### 7.1 Permitted Exceptions

| Context | Waivable Requirements |
|---------|----------------------|
| Prototype/spike | Coverage threshold |
| One-off script | Full CI |
| Vendor/generated code | All lint rules |

### 7.2 Exception Documentation

```python
# type: ignore[arg-type]  # Legacy API constraint
# noqa: E501  # URL cannot be split
# pragma: no cover  # Production-only path
```

---

## 8. References

- [GUIDELINES.md](GUIDELINES.md) — Workflows and processes
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Conventional Comments](https://conventionalcomments.org/)
- [Conventional Branch](https://conventional-branch.github.io/)
