"""M-1..M-9 patches. Run: python tools/apply_m_fixes.py"""
import ast

def _apply(path, replacements, syntax_check=True):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    for i, (old, new) in enumerate(replacements):
        n = src.count(old)
        if n == 0:
            print(f"  SKIP fix #{i+1} (already applied or not found)")
            continue
        src = src.replace(old, new)
        print(f"  fix #{i+1} OK ({n} occurrence(s))")
    if syntax_check:
        ast.parse(src)
        print(f"  syntax OK ({len(src)} bytes)")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return src


PROJECT_FIXES = [
    # M-1: entry point by depends_on structure
    (
        'def _file_likely_entry_point(file_entry: dict) -> bool:\n'
        '    """\u0424\u0430\u0439\u043b \u0432\u044b\u0433\u043b\u044f\u0434\u0438\u0442 \u043a\u0430\u043a entry point: \u0431\u0430\u0437\u043e\u0432\u043e\u0435 \u0438\u043c\u044f main.py/__main__.py/app.py/run.py/cli.py.\n'
        '    \u041d\u0435 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442 \u043a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u043b\u043e\u0432\u0430 \u0438\u0437 \u0437\u0430\u0434\u0430\u0447\u0438 \u2014 \u0442\u043e\u043b\u044c\u043a\u043e \u0438\u043c\u044f \u0444\u0430\u0439\u043b\u0430."""\n'
        '    if not isinstance(file_entry, dict):\n'
        '        return False\n'
        '    rel = _norm_path(file_entry.get("path") or "")\n'
        '    base = rel.rsplit("/", 1)[-1].lower()\n'
        '    return base in _LIKELY_ENTRY_BASENAMES',
        'def _file_likely_entry_point(file_entry: dict) -> bool:\n'
        '    """M-1 fix: entry point by name OR empty depends_on."""\n'
        '    if not isinstance(file_entry, dict):\n'
        '        return False\n'
        '    rel = _norm_path(file_entry.get("path") or "")\n'
        '    base = rel.rsplit("/", 1)[-1].lower()\n'
        '    if base in _LIKELY_ENTRY_BASENAMES:\n'
        '        return True\n'
        '    deps = file_entry.get("depends_on") or []\n'
        '    if isinstance(deps, list):\n'
        '        real_deps = [d for d in deps if isinstance(d, str) and d.lower() != "stdlib"]\n'
        '        if not real_deps:\n'
        '            return True\n'
        '    return False'
    ),
    # M-2: input file not yet on disk
    (
        '        if not _is_python_path(dep_norm):\n'
        '            if real_path.is_file():\n'
        '                read_only_paths.append(str(real_path))\n'
        '            continue',
        '        if not _is_python_path(dep_norm):\n'
        '            if real_path.is_file():\n'
        '                read_only_paths.append(str(real_path))\n'
        '            elif dep_norm in (plan.get("inputs") or []):  # M-2\n'
        '                neighbor_descs.append(f"# input (not yet on disk): {dep_norm}")\n'
        '            continue'
    ),
    # M-3: log VCS deps
    (
        'def _parse_requirements_txt(content: str) -> list[str]:\n'
        '    """\u0420\u0430\u0437\u043e\u0431\u0440\u0430\u0442\u044c requirements.txt \u2014 \u043f\u043e \u043e\u0434\u043d\u043e\u043c\u0443 \u043f\u0430\u043a\u0435\u0442\u0443 \u043d\u0430 \u0441\u0442\u0440\u043e\u043a\u0443, \u0438\u0433\u043d\u043e\u0440\u0438\u0440\u0443\u044f # \u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u0438."""\n'
        '    pkgs: list[str] = []\n'
        '    for raw in (content or "").splitlines():\n'
        '        line = raw.split("#", 1)[0].strip()\n'
        '        if not line:\n'
        '            continue\n'
        '        # \u041f\u0440\u043e\u043f\u0443\u0441\u043a\u0430\u0435\u043c \u0441\u0442\u0440\u043e\u043a\u0438 \u0432\u0438\u0434\u0430 -r other.txt, -e ., \u0438 \u043f\u0440\u043e\u0447\u0443\u044e \u0444\u043b\u0430\u0433\u043e\u0432\u0443\u044e \u043c\u0443\u0442\u044c\n'
        '        if line.startswith("-") or line in _PKG_STOPWORDS:\n'
        '            continue\n'
        '        if _PKG_PATTERN.match(line):\n'
        '            pkgs.append(line)\n'
        '    return pkgs',
        'def _parse_requirements_txt(content: str) -> list[str]:\n'
        '    """M-3 fix: log VCS/URL deps via logger.debug."""\n'
        '    pkgs: list[str] = []\n'
        '    for raw in (content or "").splitlines():\n'
        '        line = raw.split("#", 1)[0].strip()\n'
        '        if not line:\n'
        '            continue\n'
        '        if line.startswith("-") or line in _PKG_STOPWORDS:\n'
        '            continue\n'
        '        if _PKG_PATTERN.match(line):\n'
        '            pkgs.append(line)\n'
        '        elif "@" in line or line.startswith("git+") or line.startswith("http"):\n'
        '            logger.debug(f"[requirements] VCS/URL skipped: {line!r}")\n'
        '    return pkgs'
    ),
    # M-4: import→pip map
    (
        '_IMPORT_TO_PIP = {\n'
        '    "cv2": "opencv-python",\n'
        '    "PIL": "Pillow",\n'
        '    "yaml": "PyYAML",\n'
        '    "sklearn": "scikit-learn",\n'
        '    "bs4": "beautifulsoup4",\n'
        '    "dateutil": "python-dateutil",\n'
        '    "dotenv": "python-dotenv",\n'
        '}',
        '_IMPORT_TO_PIP = {\n'
        '    "cv2": "opencv-python",\n'
        '    "PIL": "Pillow",\n'
        '    "yaml": "PyYAML",\n'
        '    "sklearn": "scikit-learn",\n'
        '    "bs4": "beautifulsoup4",\n'
        '    "dateutil": "python-dateutil",\n'
        '    "dotenv": "python-dotenv",\n'
        '    # M-4 fix\n'
        '    "jwt": "PyJWT",\n'
        '    "magic": "python-magic",\n'
        '    "wx": "wxPython",\n'
        '    "usb": "pyusb",\n'
        '    "serial": "pyserial",\n'
        '    "Crypto": "pycryptodome",\n'
        '    "OpenSSL": "pyOpenSSL",\n'
        '    "nacl": "PyNaCl",\n'
        '}'
    ),
    # M-5: html fixtures
    (
        '    ".yaml": (\n'
        '        "items:\n"\n'
        '        "  - id: 1\n    name: Alice\n    email: alice@example.com\n"\n'
        '        "  - id: 2\n    name: Bob\n    email: bob@test.org\n"\n'
        '    ),\n'
        '}',
        '    ".yaml": (\n'
        '        "items:\n"\n'
        '        "  - id: 1\n    name: Alice\n    email: alice@example.com\n"\n'
        '        "  - id: 2\n    name: Bob\n    email: bob@test.org\n"\n'
        '    ),\n'
        '    # M-5 fix: html fixtures\n'
        '    ".html": (\n'
        '        "<!DOCTYPE html>\n"\n'
        '        "<html><head><title>Sample</title></head>"\n'
        '        "<body><h1>Hello</h1>"\n'
        '        "<p>Email: info@example.com</p>"\n'
        '        "</body></html>\n"\n'
        '    ),\n'
        '    ".htm": "<!DOCTYPE html>\n<html><body><p>Test</p></body></html>\n",\n'
        '}'
    ),
    # M-7: async_function kind normalization
    (
        '        expected_kind = (exp.get("kind") or "").strip().lower()\n'
        '        actual_kind = found_kinds[name]\n'
        '        if expected_kind and expected_kind != actual_kind:\n'
        '            out["kind_mismatch"].append({\n'
        '                "name": name,\n'
        '                "expected": expected_kind,\n'
        '                "actual": actual_kind,\n'
        '            })',
        '        expected_kind = (exp.get("kind") or "").strip().lower()\n'
        '        actual_kind = found_kinds[name]\n'
        '        # M-7 fix: async_function in plan == function in AST\n'
        '        _ek = "function" if expected_kind == "async_function" else expected_kind\n'
        '        if _ek and _ek != actual_kind:\n'
        '            out["kind_mismatch"].append({\n'
        '                "name": name,\n'
        '                "expected": expected_kind,\n'
        '                "actual": actual_kind,\n'
        '            })'
    ),
    # M-8: class stub with __init__
    (
        '            lines.append(f"# contract: {sig}")\n'
        '            lines.append(f"class {name}:")\n'
        '            ds = doc or sig\n'
        '            if ds:\n'
        "                lines.append(f'    \"{\"\"\"}{ds}{\"\"\"}'\n"
        '            lines.append("    pass")\n'
        '            lines.append("")',
        '            # M-8 fix: class stub with __init__\n'
        '            lines.append(f"class {name}:")\n'
        '            ds = doc or sig\n'
        '            if ds:\n'
        "                lines.append(f'    \"{\"\"\"}{ds}{\"\"\"}'\n"
        '            _isig = (exp.get("signature") or "").strip()\n'
        '            if _isig and "(" in _isig and not _isig.startswith("class"):\n'
        '                _ap = _isig[_isig.index("("):]\n'
        '                _inner = _ap[1:].rstrip(")")\n'
        '                _sep = ", " if _inner.strip() else ""\n'
        '                lines.append(f"    def __init__(self{_sep}{_inner}): ...")\n'
        '                lines.append(f"    # contract: {_isig}")\n'
        '            else:\n'
        '                lines.append("    pass")\n'
        '            lines.append("")'
    ),
]

PROJECTS_FIXES = [
    # M-6: no timestamp for resume
    (
        '    # P9.6: \u0432\u0441\u0435\u0433\u0434\u0430 \u0434\u043e\u0431\u0430\u0432\u043b\u044f\u0435\u043c timestamp \u0434\u043b\u044f \u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0438 \u0443\u043d\u0438\u043a\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438\n'
        '    # \u0431\u044b\u043b\u043e: \u043f\u0440\u043e\u0432\u0435\u0440\u044f\u043b\u0430\u0441\u044c \u0442\u043e\u043b\u044c\u043a\u043e \u0444\u0430\u0439\u043b\u043e\u0432\u0430\u044f \u0441\u0438\u0441\u0442\u0435\u043c\u0430, \u043d\u043e \u0435\u0441\u043b\u0438 \u043f\u0440\u0435\u0434\u044b\u0434\u0443\u0449\u0438\u0439 \u043f\u0440\u043e\u0435\u043a\u0442 \u0431\u044b\u043b \u0443\u0434\u0430\u043b\u0451\u043d \u0438\u043b\u0438 \u0435\u0449\u0451 \u043d\u0435 \u0437\u0430\u043a\u043e\u043c\u043c\u0438\u0447\u0435\u043d \u043d\u0430 \u0434\u0438\u0441\u043a, \u0431\u0443\u0434\u0435\u0442 \u043a\u043e\u043b\u043b\u0438\u0437\u0438\u044f.\n'
        '    # \u0418\u0441\u043f\u0440\u0430\u0432\u043b\u044f\u0435\u0442 nightly E2E \u0431\u0430\u0433: csv_to_json \u0438 rename_files \u043f\u0438\u0441\u0430\u043b\u0438 \u0432 \u043e\u0434\u0438\u043d slug.\n'
        '    ts_suffix = str(int(time.time()))\n'
        '    base_max = MAX_SLUG_LEN - len(ts_suffix) - 1\n'
        '    slug = f"{slug[:base_max]}-{ts_suffix}"',
        '    # P9.6 / M-6 fix: no timestamp for resume\n'
        '    if not spec.get("_resume"):\n'
        '        ts_suffix = str(int(time.time()))\n'
        '        base_max = MAX_SLUG_LEN - len(ts_suffix) - 1\n'
        '        slug = f"{slug[:base_max]}-{ts_suffix}"'
    ),
    # M-9a: extract constant
    (
        '# P11.4: \u043f\u043e\u0434\u043d\u044f\u0442\u0430 \u043e\u0431\u0440\u0435\u0437\u043a\u0430 \u0441 300 \u0434\u043e 4000 \u2014 \u0438\u043d\u0430\u0447\u0435 \u0432 jsonl \u0442\u0435\u0440\u044f\u044e\u0442\u0441\u044f \u043f\u043e\u043b\u044f \u0432\u0440\u043e\u0434\u0435 contract.missing,\n',
        '# P11.4: \u043f\u043e\u0434\u043d\u044f\u0442\u0430 \u043e\u0431\u0440\u0435\u0437\u043a\u0430 \u0441 300 \u0434\u043e 4000 \u2014 \u0438\u043d\u0430\u0447\u0435 \u0432 jsonl \u0442\u0435\u0440\u044f\u044e\u0442\u0441\u044f \u043f\u043e\u043b\u044f \u0432\u0440\u043e\u0434\u0435 contract.missing,\n'
        '# M-9 fix\n'
        '_OUTPUT_TRUNCATE_BYTES = 4000\n'
    ),
    # M-9b: use constant
    (
        '        "stdout": out[-4000:],\n        "stderr": err[-4000:],',
        '        "stdout": out[-_OUTPUT_TRUNCATE_BYTES:],\n        "stderr": err[-_OUTPUT_TRUNCATE_BYTES:],'
    ),
]


print("=== brain/agents/project.py ===")
_apply("brain/agents/project.py", PROJECT_FIXES)
print()
print("=== tools/projects.py ===")
_apply("tools/projects.py", PROJECTS_FIXES)
print()
print("Done. Run:")
print("  git add brain/agents/project.py tools/projects.py")
print("  git commit -m 'fix(M-1..M-9)'")
print("  git push origin main")
