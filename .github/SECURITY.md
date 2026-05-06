# Security Policy

---

## Supported Versions

| Version | Supported |
|:---|:---|
| 1.0.x | ✅ Active |
| < 1.0 | ❌ Not supported |

---

## Scope

MindSight is an academic research pipeline that processes publicly available survey data.
It contains no authentication systems, no external API integrations, no user data collection, and no production infrastructure.

Security disclosures relevant to this project include:

- Malicious content embedded in CSV data files that could execute during pandas parsing.
- Dependency vulnerabilities in `numpy`, `pandas`, `nbconvert`, or `jupyter`.
- Unsafe deserialization in `.ipynb` files loaded by the execution engine.

---

## Reporting a Vulnerability

Do not open a public GitHub issue for security vulnerabilities.

**Report privately:**

1. Navigate to the repository's **Security** tab on GitHub.
2. Select **Report a vulnerability** (GitHub Private Security Advisory).
3. Include: description, reproduction steps, affected version, and potential impact.

**Response timeline:**

| Action | Target Time |
|:---|:---|
| Acknowledgement | 48 hours |
| Triage and severity assessment | 5 business days |
| Patch or mitigation | 14 business days (critical) / 30 days (medium/low) |
| Public disclosure | After patch is released and users have had time to update |

---

## Dependency Auditing

Audit dependencies for known CVEs using:

```bash
pip audit
```

Pin all dependencies to exact versions in `requirements.txt` for reproducible, auditable builds.
