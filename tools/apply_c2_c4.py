"""One-shot patch script: C-2 + C-4 fixes for brain/agents/project.py.

Run from repo root:
  python tools/apply_c2_c4.py
"""
import ast
import urllib.request

TARGET = "brain/agents/project.py"
ORIGIN_URL = (
    "https://raw.githubusercontent.com/s4s4s4s/Jarvis/"
    "50ade11fa0f4e4ef775374cf2e0e9187b0671855/brain/agents/project.py"
)

# ------------------------------------------------------------------ fetch
print("Reading", TARGET, "...")
with open(TARGET, encoding="utf-8") as f:
    src = f.read()

if len(src) < 50_000:
    print("Local file looks truncated (", len(src), "bytes). Fetching original from GitHub...")
    with urllib.request.urlopen(ORIGIN_URL) as r:
        src = r.read().decode("utf-8")
    print("Fetched", len(src), "bytes")

# ------------------------------------------------------------------ C-2
OLD_C2 = (
    "    # P11.7 FIX-2: \u0445\u0435\u0448\u0438 \u0444\u0430\u0439\u043b\u043e\u0432 \u0434\u043e heal-\u0438\u0442\u0435\u0440\u0430\u0446\u0438\u0438 \u0434\u043b\u044f no-change guard\n"
    "    import hashlib as hashlib  "
    "# \u0443\u0436\u0435 \u0438\u043c\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u043d \u0447\u0435\u0440\u0435\u0437 hashlib \u0432\u044b\u0448\u0435, \u043d\u043e \u044f\u0432\u043d\u043e \u0434\u043b\u044f \u044f\u0441\u043d\u043e\u0441\u0442\u0438\n"
    "    _pre_hashes: dict[str, str] = {}"
)
NEW_C2 = (
    "    # P11.7 FIX-2: \u0445\u0435\u0448\u0438 \u0444\u0430\u0439\u043b\u043e\u0432 \u0434\u043e heal-\u0438\u0442\u0435\u0440\u0430\u0446\u0438\u0438 \u0434\u043b\u044f no-change guard\n"
    "    # hashlib \u0438\u043c\u043f\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u0430\u043d \u043d\u0430 \u0432\u0435\u0440\u0445\u043d\u0435\u043c \u0443\u0440\u043e\u0432\u043d\u0435 \u043c\u043e\u0434\u0443\u043b\u044f (C-2 fix: \u0443\u0431\u0440\u0430\u043d \u0434\u0443\u0431\u043b\u0438\u0440\u0443\u044e\u0449\u0438\u0439 import)\n"
    "    _pre_hashes: dict[str, str] = {}"
)
assert OLD_C2 in src, "C-2 pattern not found!"
src = src.replace(OLD_C2, NEW_C2)
print("C-2 patched OK")

# ------------------------------------------------------------------ C-4
OLD_C4 = (
    '        "iters":   res.attempts,\n'
    '        "static":  static_summary,\n'
    '        "contract": contract_check,  # P11.2.d\n'
    '        "contract_failure": contract_failed,  # P11.5.C: \u044f\u0432\u043d\u044b\u0439 \u0444\u043b\u0430\u0433 \u0434\u043b\u044f heal-loop\n'
    '        "_via":    "aider",\n'
    '        "aider":   {\n'
    '            "duration_s": res.duration_s,\n'
    '            "exit_code":  res.exit_code,\n'
    '        },\n'
    '    }'
)
NEW_C4 = (
    '        # C-4 fix: \u043f\u043e\u043b\u0435 "error" \u043f\u0440\u0438 contract_failed — \u043d\u0443\u0436\u043d\u043e _diagnose \u0438 heal-loop\n'
    '        "error": (\n'
    '            f"contract violated: missing={contract_check.get(\'missing\') or []} "\n'
    '            f"kind_mismatch={contract_check.get(\'kind_mismatch\') or []}"\n'
    '            if contract_failed else ""\n'
    '        ),\n'
    '        "iters":   res.attempts,\n'
    '        "static":  static_summary,\n'
    '        "contract": contract_check,  # P11.2.d\n'
    '        "contract_failure": contract_failed,  # P11.5.C\n'
    '        "_via":    "aider",\n'
    '        "aider":   {\n'
    '            "duration_s": res.duration_s,\n'
    '            "exit_code":  res.exit_code,\n'
    '        },\n'
    '    }'
)
assert OLD_C4 in src, "C-4 pattern not found!"
src = src.replace(OLD_C4, NEW_C4)
print("C-4 patched OK")

# ------------------------------------------------------------------ validate
ast.parse(src)
print("Syntax OK")

# ------------------------------------------------------------------ write
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(src)
print(f"Written {len(src)} bytes, {src.count(chr(10))} lines")
print("Done! Now run: git add brain/agents/project.py && git commit -m 'fix(C-2,C-4)' && git push")
