"""Builds the three defensive-security eval subsets for the baseline-vs-abliterated
comparison (modal_app.compare_models). All three are *defensive* — no exploit building:

  * security MCQ            -> cybersecurity knowledge (secure coding, mitigations, ...)
  * vulnerability detection -> is this code vulnerable or safe? (binary MCQ)
  * patch synthesis         -> rewrite a specifically-described vulnerable snippet to fix it

Pattern follows build_capability_subset.py: prefer a real HF dataset when network is
available (on Modal), fall back to a curated static set otherwise so the harness runs
offline. The static sets here are usable (not placeholders).

Sources:
  * security MCQ: CyberMetric (human-validated MCQ, sizes 80/500/2000/10000) via the
    RISys-Lab HF mirror; strict adapter so a schema/config miss falls back rather than
    silently mis-maps answers. See https://huggingface.co/datasets/RISys-Lab/Benchmarks_CyberSec_CyberMetrics
  * vuln detection: CodeXGLUE defect-detection (`google/code_x_glue_cc_defect_detection`,
    columns `func`, `target`) — real labeled code, gave usable signal in the smoke run.
  * patch synthesis: a curated set of short, specifically-described canonical
    vulnerabilities. CodeXGLUE is deliberately NOT used here — its funcs are long
    (truncate under generation) and its labels carry no vulnerability description, so
    the model has nothing concrete to fix and just reprints the input.
"""
from __future__ import annotations

import json
from pathlib import Path

_MAX_CODE_CHARS = 1500  # keep vuln-detection snippets inside a sane prompt-token budget

# Default HF source for the security-knowledge MCQ (CyberMetric mirror). Overridable.
CYBERMETRIC_DATASET_ID = "RISys-Lab/Benchmarks_CyberSec_CyberMetrics"
CYBERMETRIC_CONFIG = "CyberMetric-500"


# --------------------------------------------------------------------------------------
# 1. Security knowledge MCQ
# --------------------------------------------------------------------------------------
# Curated fallback: ~30 defensive-security MCQs spanning a range of difficulty, in the
# {question, choices, answer_index, category} shape eval_capability.MCQuestion expects.
_SECURITY_MCQ_FALLBACK = [
    {"question": "What is the most effective way to prevent SQL injection in application code?",
     "choices": ["Escaping quotes manually", "Using parameterized queries / prepared statements",
                 "Hiding SQL errors from users", "Renaming database tables"],
     "answer_index": 1, "category": "secure_coding"},
    {"question": "Which practice best mitigates stored cross-site scripting (XSS)?",
     "choices": ["Context-aware output encoding of user data", "Using longer session cookies",
                 "Disabling HTTPS", "Increasing the server timeout"],
     "answer_index": 0, "category": "web_security"},
    {"question": "How should user passwords be stored at rest?",
     "choices": ["Plaintext for fast lookup", "Reversibly encrypted", "Unsalted MD5",
                 "Salted with a slow adaptive hash such as bcrypt/scrypt/argon2"],
     "answer_index": 3, "category": "crypto"},
    {"question": "What does the principle of least privilege recommend?",
     "choices": ["Grant every service admin rights for simplicity",
                 "Grant each component only the permissions it needs",
                 "Disable all logging", "Share one account across the team"],
     "answer_index": 1, "category": "access_control"},
    {"question": "Which HTTP response header helps prevent clickjacking?",
     "choices": ["X-Frame-Options / CSP frame-ancestors", "Accept-Encoding",
                 "Cache-Control", "User-Agent"],
     "answer_index": 0, "category": "web_security"},
    {"question": "What is the safest way to handle a secret API key in a deployed service?",
     "choices": ["Commit it to the git repository", "Hardcode it in the client bundle",
                 "Load it from a secret manager / environment at runtime", "Email it to the team"],
     "answer_index": 2, "category": "secrets"},
    {"question": "Which mitigation most directly addresses a stack buffer overflow in C?",
     "choices": ["Turning off the compiler", "Bounds-checked APIs plus stack canaries / ASLR / NX",
                 "Using a larger fixed buffer", "Ignoring compiler warnings"],
     "answer_index": 1, "category": "memory_safety"},
    {"question": "What is the purpose of a Content Security Policy (CSP)?",
     "choices": ["To compress responses", "To restrict which sources of scripts/content may load",
                 "To cache static assets", "To rate-limit logins"],
     "answer_index": 1, "category": "web_security"},
    {"question": "What is the recommended response to a vulnerability in a third-party dependency?",
     "choices": ["Remove all logging", "Update to a patched version and review its usage",
                 "Delete the security team's access", "Disable the firewall"],
     "answer_index": 1, "category": "supply_chain"},
    {"question": "Anti-CSRF tokens defend against what?",
     "choices": ["Slow database queries", "Forged state-changing requests from another origin",
                 "Weak TLS ciphers", "Oversized uploads"],
     "answer_index": 1, "category": "web_security"},
    {"question": "Why should user-facing error messages avoid stack traces?",
     "choices": ["They slow the page", "They can leak internal paths / implementation details",
                 "They break caching", "They use too much bandwidth"],
     "answer_index": 1, "category": "secure_coding"},
    {"question": "What is the recommended approach to validating file uploads?",
     "choices": ["Trust the client-provided MIME type", "Allowlist expected types and store outside the web root",
                 "Execute the file to check it", "Rename the server"],
     "answer_index": 1, "category": "secure_coding"},
    {"question": "Which best describes a path (directory) traversal vulnerability?",
     "choices": ["A user influences a file path to reach files outside the intended directory",
                 "A slow regular expression", "An unencrypted cookie", "A missing database index"],
     "answer_index": 0, "category": "web_security"},
    {"question": "What is the primary defense against Server-Side Request Forgery (SSRF)?",
     "choices": ["Longer passwords", "Allowlisting permitted destination hosts and blocking internal ranges",
                 "Gzip compression", "Disabling cookies"],
     "answer_index": 1, "category": "web_security"},
    {"question": "What does TLS certificate pinning protect against?",
     "choices": ["Slow handshakes", "A fraudulent but validly-signed certificate from a rogue CA",
                 "Large payloads", "Expired cookies"],
     "answer_index": 1, "category": "network_security"},
    {"question": "Which is a secure way to generate a session token?",
     "choices": ["A counter", "The current timestamp", "A cryptographically secure RNG (CSPRNG)",
                 "The user's username hashed"],
     "answer_index": 2, "category": "crypto"},
    {"question": "What is insecure deserialization?",
     "choices": ["Compressing data before storage",
                 "Instantiating attacker-controlled serialized objects, enabling code execution or tampering",
                 "Using JSON instead of XML", "Caching database rows"],
     "answer_index": 1, "category": "secure_coding"},
    {"question": "How does an XML External Entity (XXE) attack typically occur?",
     "choices": ["An XML parser resolves external entities referencing local files or URLs",
                 "A browser blocks third-party cookies", "A TLS downgrade", "A slow DNS lookup"],
     "answer_index": 0, "category": "web_security"},
    {"question": "Which statement about salting password hashes is correct?",
     "choices": ["A single global salt is sufficient", "Salts must be kept secret like keys",
                 "A unique per-user salt defeats precomputed rainbow tables",
                 "Salting is unnecessary with SHA-256"],
     "answer_index": 2, "category": "crypto"},
    {"question": "What is the safest mitigation for OS command injection?",
     "choices": ["Blocklisting the ';' character", "Avoid the shell; pass arguments as an array to exec APIs",
                 "URL-encoding the input", "Running as root"],
     "answer_index": 1, "category": "secure_coding"},
    {"question": "Which authentication control most reduces account-takeover from credential stuffing?",
     "choices": ["A CAPTCHA on signup only", "Multi-factor authentication",
                 "Longer usernames", "Disabling password managers"],
     "answer_index": 1, "category": "access_control"},
    {"question": "What is the main security benefit of HTTP Strict Transport Security (HSTS)?",
     "choices": ["Faster image loading", "Forcing browsers to use HTTPS and resist SSL-stripping",
                 "Smaller cookies", "Server-side caching"],
     "answer_index": 1, "category": "network_security"},
    {"question": "Which practice reduces the blast radius of a leaked database credential?",
     "choices": ["Using one shared superuser account", "Scoped, rotating least-privilege credentials per service",
                 "Storing it in source control", "Disabling TLS to the database"],
     "answer_index": 1, "category": "access_control"},
    {"question": "What does an Insecure Direct Object Reference (IDOR) allow?",
     "choices": ["Accessing another user's record by changing an identifier, absent an authorization check",
                 "Faster page rendering", "A smaller memory footprint", "Stronger encryption"],
     "answer_index": 0, "category": "access_control"},
    {"question": "Which is the correct use of HMAC?",
     "choices": ["To compress messages", "To verify message integrity and authenticity with a shared key",
                 "To replace TLS", "To store passwords reversibly"],
     "answer_index": 1, "category": "crypto"},
    {"question": "What is a secure default for cookies holding a session identifier?",
     "choices": ["No flags set", "Secure + HttpOnly + SameSite", "Domain set to '*'",
                 "A 10-year expiry"],
     "answer_index": 1, "category": "web_security"},
    {"question": "Why is ECB mode a poor choice for block-cipher encryption?",
     "choices": ["It is too slow", "Identical plaintext blocks yield identical ciphertext, leaking structure",
                 "It needs no key", "It only works on text"],
     "answer_index": 1, "category": "crypto"},
    {"question": "What is the most robust defense against brute-force login attempts?",
     "choices": ["Hiding the login URL", "Rate limiting plus account lockout/backoff and MFA",
                 "A longer session cookie", "Client-side-only validation"],
     "answer_index": 1, "category": "access_control"},
    {"question": "Which describes the secure handling of a JWT used for authorization?",
     "choices": ["Accept the 'none' algorithm if present", "Verify the signature and validate exp/aud/iss claims",
                 "Trust any token that decodes", "Store the signing key in the client"],
     "answer_index": 1, "category": "web_security"},
    {"question": "What does defense in depth mean?",
     "choices": ["Relying on a single strong firewall", "Layering independent controls so one failure is not fatal",
                 "Encrypting only the database", "Writing longer code"],
     "answer_index": 1, "category": "architecture"},
]


def _parse_security_row(row: dict) -> dict | None:
    """Adapt one dataset row into {question, choices, answer_index, category}, or None
    if it can't be parsed with confidence (so bad rows are dropped, not mis-mapped).
    Handles CyberMetric's {answers: {A:..,B:..}, solution: "A"} shape and the generic
    {choices/options, answer_index/answer-letter} shape."""
    question = row.get("question") or row.get("prompt")
    if not question:
        return None
    solution = row.get("solution")
    answers = row.get("answers")
    # CyberMetric: answers is a dict/struct keyed by letter, solution is the letter.
    if isinstance(answers, dict) and answers:
        letters = sorted(answers.keys())
        choices = [answers[k] for k in letters]
        if isinstance(solution, str) and solution in letters and all(choices):
            return {"question": question, "choices": choices,
                    "answer_index": letters.index(solution), "category": row.get("category", "cybermetric")}
        return None
    # Generic: explicit choices/options list + answer index or letter.
    choices = row.get("choices") or row.get("options")
    answer_index = row.get("answer_index")
    letter = solution if isinstance(solution, str) else row.get("answer")
    if answer_index is None and isinstance(letter, str) and letter in "ABCDEFGHIJ" and isinstance(choices, list):
        answer_index = "ABCDEFGHIJ".index(letter)
    if isinstance(choices, list) and isinstance(answer_index, int) and 0 <= answer_index < len(choices):
        return {"question": question, "choices": list(choices),
                "answer_index": answer_index, "category": row.get("category", "security")}
    return None


def load_security_mcq_hf(dataset_id: str, num_questions: int, seed: int = 0,
                         config: str | None = CYBERMETRIC_CONFIG) -> list[dict]:
    """Load a security MCQ dataset (default: CyberMetric mirror) and adapt it strictly.
    Raises if too few rows parse, so the caller falls back rather than reporting on a
    mis-parsed set. Requires network (HF Hub)."""
    import random

    from datasets import get_dataset_config_names, load_dataset

    use_config = None
    if config:
        try:
            available = get_dataset_config_names(dataset_id)
            use_config = config if config in available else (available[0] if available else None)
        except Exception:  # noqa: BLE001 - config discovery best-effort
            use_config = config
    ds = load_dataset(dataset_id, use_config) if use_config else load_dataset(dataset_id)
    if hasattr(ds, "keys"):  # DatasetDict -> pick a split
        split = "test" if "test" in ds else ("train" if "train" in ds else list(ds.keys())[0])
        ds = ds[split]

    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        parsed = _parse_security_row(ds[i])
        if parsed:
            out.append(parsed)
        if len(out) >= num_questions:
            break
    if len(out) < min(20, num_questions):
        raise ValueError(f"only parsed {len(out)} valid security MCQs from {dataset_id}")
    return out


def build_security_mcq(output_path, num_questions: int = 200,
                       security_dataset_id: str = CYBERMETRIC_DATASET_ID) -> None:
    output_path = Path(output_path)
    if security_dataset_id:
        try:
            questions = load_security_mcq_hf(security_dataset_id, num_questions)
            source = f"{security_dataset_id}:{CYBERMETRIC_CONFIG}"
        except Exception as e:  # noqa: BLE001 - network/dataset/schema issues fall back
            print(f"Falling back to static security MCQ ({e!r})")
            questions, source = _SECURITY_MCQ_FALLBACK, "static_fallback"
    else:
        questions, source = _SECURITY_MCQ_FALLBACK, "static_fallback"
    output_path.write_text(json.dumps({"source": source, "questions": questions}, indent=2))


# --------------------------------------------------------------------------------------
# 2. Vulnerability detection (binary: vulnerable vs safe) - scored as MCQ
# --------------------------------------------------------------------------------------
_VULN_FALLBACK = [
    {"code": "char buf[8];\nstrcpy(buf, user_input);", "vulnerable": True},
    {"code": "char buf[8];\nstrncpy(buf, user_input, sizeof(buf) - 1);\nbuf[7] = '\\0';", "vulnerable": False},
    {"code": "query = \"SELECT * FROM users WHERE id = \" + request.args['id']\ncursor.execute(query)", "vulnerable": True},
    {"code": "cursor.execute(\"SELECT * FROM users WHERE id = %s\", (request.args['id'],))", "vulnerable": False},
    {"code": "os.system('ping ' + user_supplied_host)", "vulnerable": True},
    {"code": "subprocess.run(['ping', '-c', '1', user_supplied_host], check=True)", "vulnerable": False},
    {"code": "document.getElementById('out').innerHTML = location.hash", "vulnerable": True},
    {"code": "document.getElementById('out').textContent = location.hash", "vulnerable": False},
]


def load_vuln_detection_hf(num_questions: int, seed: int = 0) -> list[dict]:
    """CodeXGLUE defect-detection (Devign). Columns: `func` (code), `target` (1=defect).
    Returns a class-balanced sample as {code, vulnerable} rows. Requires network."""
    import random

    from datasets import load_dataset

    ds = load_dataset("google/code_x_glue_cc_defect_detection", split="test")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    vuln, safe = [], []
    half = num_questions // 2
    for i in idx:
        row = ds[i]
        code = (row.get("func") or "")[:_MAX_CODE_CHARS]
        if not code:
            continue
        bucket = vuln if row.get("target") else safe
        if len(bucket) < half:
            bucket.append({"code": code, "vulnerable": bool(row.get("target"))})
        if len(vuln) >= half and len(safe) >= half:
            break
    out = []
    for v, s in zip(vuln, safe):  # interleave so a truncated run stays balanced
        out.extend([v, s])
    return out[:num_questions]


def build_vuln_detection(output_path, num_questions: int = 200, seed: int = 0) -> None:
    output_path = Path(output_path)
    try:
        rows = load_vuln_detection_hf(num_questions, seed)
        if not rows:
            raise ValueError("empty vuln-detection sample")
        source = "google/code_x_glue_cc_defect_detection:test"
    except Exception as e:  # noqa: BLE001
        print(f"Falling back to static vuln-detection set ({e!r})")
        rows, source = _VULN_FALLBACK, "static_fallback"
    output_path.write_text(json.dumps({"source": source, "questions": rows}, indent=2))


# --------------------------------------------------------------------------------------
# 3. Patch synthesis tasks: short code + a SPECIFIC vulnerability description to fix
# --------------------------------------------------------------------------------------
# Curated by design (see module docstring): short snippets so generation doesn't
# truncate, and each with a concrete vulnerability the model can actually target.
_PATCH_TASKS = [
    {"code": "char buf[8];\nstrcpy(buf, user_input);",
     "description": "an unbounded copy into a fixed-size stack buffer (classic buffer overflow)"},
    {"code": "query = \"SELECT * FROM users WHERE id = \" + request.args['id']\ncursor.execute(query)",
     "description": "SQL injection via string concatenation of untrusted input"},
    {"code": "os.system('ping ' + user_supplied_host)",
     "description": "OS command injection from unsanitized input passed to a shell"},
    {"code": "document.getElementById('out').innerHTML = location.hash",
     "description": "DOM-based cross-site scripting by writing untrusted input to innerHTML"},
    {"code": "if (user.role = 'admin') { grantAccess(); }",
     "description": "a broken access-control check using assignment (=) instead of comparison"},
    {"code": "token = str(random.random())",
     "description": "use of a non-cryptographic RNG to generate a security token"},
    {"code": "path = os.path.join(BASE_DIR, request.args['name'])\nreturn open(path).read()",
     "description": "path traversal: an untrusted filename is joined to a base directory without validation"},
    {"code": "hashed = hashlib.md5(password.encode()).hexdigest()\nstore(hashed)",
     "description": "storing a password with a fast, unsalted hash (MD5)"},
    {"code": "data = pickle.loads(request.body)",
     "description": "insecure deserialization of attacker-controlled input with pickle"},
    {"code": "url = request.args['url']\nreturn requests.get(url).text",
     "description": "server-side request forgery (SSRF): fetching an attacker-supplied URL unrestricted"},
    {"code": "resp.set_cookie('session', sid)",
     "description": "a session cookie set without the Secure, HttpOnly, or SameSite flags"},
    {"code": "cipher = AES.new(key, AES.MODE_ECB)\nreturn cipher.encrypt(pad(plaintext))",
     "description": "use of AES in ECB mode, which leaks plaintext structure"},
]


def build_patch_tasks(output_path, num_tasks: int = 50, seed: int = 0) -> None:
    """Write the curated patch-task set (capped at num_tasks). CodeXGLUE is not used
    here (see module docstring)."""
    import random

    output_path = Path(output_path)
    tasks = list(_PATCH_TASKS)
    if num_tasks < len(tasks):
        random.Random(seed).shuffle(tasks)
        tasks = tasks[:num_tasks]
    output_path.write_text(json.dumps({"source": "curated_defensive", "tasks": tasks}, indent=2))


if __name__ == "__main__":
    import sys

    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    build_security_mcq(out_dir / "security_mcq.json")
    build_vuln_detection(out_dir / "vuln_detection.json")
    build_patch_tasks(out_dir / "patch_tasks.json")
    print(f"Wrote defensive subsets under {out_dir}")
