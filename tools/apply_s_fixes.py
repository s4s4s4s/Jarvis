"""One-shot patch script: S-1 through S-7 fixes for brain/agents/project.py.

Run from repo root:
  python tools/apply_s_fixes.py
"""
import ast
import urllib.request

TARGET = "brain/agents/project.py"
BASE_COMMIT = "8105090"  # last known-good commit with C-2/C-4 applied
ORIGIN_URL = (
    f"https://raw.githubusercontent.com/s4s4s4s/Jarvis/{BASE_COMMIT}/brain/agents/project.py"
)

print("Reading", TARGET, "...")
with open(TARGET, encoding="utf-8") as f:
    src = f.read()

if len(src) < 100_000:
    print(f"Local file looks truncated ({len(src)} bytes). Fetching from GitHub...")
    with urllib.request.urlopen(ORIGIN_URL) as r:
        src = r.read().decode("utf-8")
    print(f"Fetched {len(src)} bytes")

# ------------------------------------------------------------------ S-1
OLD = (
    "    is_echo = (\n"
    "        len(reqs) == 1\n"
    "        and q_norm\n"
    "        and reqs[0].lower().startswith(q_norm[: min(40, len(q_norm))])\n"
    "    )"
)
NEW = (
    "    is_echo = (\n"
    "        len(reqs) == 1\n"
    "        and q_norm\n"
    "        # S-1 fix: strip \u043f\u0440\u043e\u0431\u0435\u043b\u043e\u0432/\u0442\u043e\u0447\u0435\u043a \u043f\u0435\u0440\u0435\u0434 \u0441\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435\u043c \u2014 LLM \u0438\u043d\u043e\u0433\u0434\u0430 \u0434\u043e\u0431\u0430\u0432\u043b\u044f\u0435\u0442 \u043f\u0440\u043e\u0431\u0435\u043b \u0438\u043b\u0438 \u0442\u043e\u0447\u043a\u0443 \u0432 \u043d\u0430\u0447\u0430\u043b\u043e\n"
    "        and reqs[0].lower().strip(\" .\").startswith(q_norm.strip(\" .\")[: min(40, len(q_norm.strip(\" .\")))]) \n"
    "    )"
)
assert OLD in src, "S-1 pattern not found!"
src = src.replace(OLD, NEW)
print("S-1 OK")

# ------------------------------------------------------------------ S-2
OLD = (
    "def budget_for_tier(tier: str) -> dict:\n"
    "    \"\"\"\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0431\u044e\u0434\u0436\u0435\u0442\u0430 \u0434\u043b\u044f \u0442\u0438\u0440\u0430. \u0414\u0435\u0444\u043e\u043b\u0442 \u2014 M.\"\"\"\n"
    "    return BUDGET_TIERS.get(tier, BUDGET_TIERS[\"M\"])"
)
NEW = (
    "def budget_for_tier(tier: str | None) -> dict:\n"
    "    \"\"\"\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0431\u044e\u0434\u0436\u0435\u0442\u0430 \u0434\u043b\u044f \u0442\u0438\u0440\u0430. \u0414\u0435\u0444\u043e\u043b\u0442 \u2014 M.\n"
    "    S-2 fix: \u043f\u0440\u0438\u043d\u0438\u043c\u0430\u0435\u0442 None \u044f\u0432\u043d\u043e \u2014 estimate_complexity \u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0440\u0443\u0435\u0442 str,\n"
    "    \u043d\u043e budget_for_tier \u043c\u043e\u0433 \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c None \u0438\u0437 \u0432\u043d\u0435\u0448\u043d\u0435\u0433\u043e \u043a\u043e\u0434\u0430.\n"
    "    \"\"\"\n"
    "    return BUDGET_TIERS.get(tier or \"M\", BUDGET_TIERS[\"M\"])"
)
assert OLD in src, "S-2 pattern not found!"
src = src.replace(OLD, NEW)
print("S-2 OK")

# ------------------------------------------------------------------ S-3
OLD = (
    "        if not new_checks and t.get(\"checks\"):\n"
    "            new_checks = [{\"type\": \"rc_zero\"}]\n"
    "            removed.append({\"check\": {\"type\": \"__all_removed__\"}, \"reason\": \"rc_zero_fallback_added\"})"
)
NEW = (
    "        if not new_checks and t.get(\"checks\"):\n"
    "            new_checks = [{\"type\": \"rc_zero\"}]\n"
    "            # S-3 fix: \u044d\u0442\u043e \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u0438\u0435 fallback, \u043d\u0435 \u0443\u0434\u0430\u043b\u0435\u043d\u0438\u0435 \u2014 \u043d\u0435 \u0437\u0430\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u043c \u0432 removed,\n"
    "            # \u0447\u0442\u043e\u0431\u044b \u043b\u043e\u0433 'removed N invalid check(s)' \u043d\u0435 \u0432\u0432\u043e\u0434\u0438\u043b \u0432 \u0437\u0430\u0431\u043b\u0443\u0436\u0434\u0435\u043d\u0438\u0435."
)
assert OLD in src, "S-3 pattern not found!"
src = src.replace(OLD, NEW)
print("S-3 OK")

# ------------------------------------------------------------------ S-4
OLD = (
    "                        _pre_hashes[_ah_target] = _hash_after\n"
    "                    except Exception:\n"
    "                        pass"
)
# Find the one inside the _ah_target block (not the init block)
idx = src.find("                        _pre_hashes[_ah_target] = _hash_after\n                    except Exception:\n                        pass")
assert idx >= 0, "S-4 pattern not found!"
NEW_SUFFIX = (
    "                        _pre_hashes[_ah_target] = _hash_after\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                    # S-4 fix: \u043e\u0431\u043d\u043e\u0432\u043b\u044f\u0435\u043c \u0445\u0435\u0448\u0438 \u0412\u0421\u0415\u0425 \u0444\u0430\u0439\u043b\u043e\u0432 \u043f\u043e\u0441\u043b\u0435 heal \u2014\n"
    "                    # \u0438\u043d\u0430\u0447\u0435 \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0430\u044f \u0438\u0442\u0435\u0440\u0430\u0446\u0438\u044f \u043c\u043e\u0436\u0435\u0442 \u043b\u043e\u0436\u043d\u043e \u0441\u0447\u0438\u0442\u0430\u0442\u044c \u0447\u0443\u0436\u043e\u0439 \u0444\u0430\u0439\u043b \u00ab\u043d\u0435 \u0438\u0437\u043c\u0435\u043d\u0438\u043b\u0441\u044f\u00bb\n"
    "                    try:\n"
    "                        for _sfp in file_paths:\n"
    "                            if _sfp == _ah_target:\n"
    "                                continue\n"
    "                            _sc = read_project_file(slug, _sfp)\n"
    "                            _pre_hashes[_sfp] = hashlib.md5(_sc.encode('utf-8', errors='replace')).hexdigest()\n"
    "                    except Exception:\n"
    "                        pass"
)
src = src[:idx] + NEW_SUFFIX + src[idx + len(OLD):]
print("S-4 OK")

# ------------------------------------------------------------------ S-5
OLD = (
    "        has_rc_check = any(c.get(\"type\") == \"rc_zero\" for c in check_results)\n"
    "        rc_implicit_ok = True if has_rc_check else (res.get(\"returncode\") == 0)\n"
    "        overall_ok = checks_ok and rc_implicit_ok"
)
NEW = (
    "        has_rc_check = any(c.get(\"type\") == \"rc_zero\" for c in check_results)\n"
    "        # S-5 fix: \u0435\u0441\u043b\u0438 runner \u0432\u0435\u0440\u043d\u0443\u043b ok=False (FileNotFoundError, venv \u043e\u0442\u0441\u0443\u0442\u0441\u0442\u0432\u0443\u0435\u0442) \u2014 rc_implicit_ok \u0442\u043e\u0436\u0435 False\n"
    "        runner_ok = bool(res.get(\"ok\", True))\n"
    "        rc_implicit_ok = runner_ok and (True if has_rc_check else (res.get(\"returncode\") == 0))\n"
    "        overall_ok = checks_ok and rc_implicit_ok"
)
assert OLD in src, "S-5 pattern not found!"
src = src.replace(OLD, NEW)
print("S-5 OK")

# ------------------------------------------------------------------ S-6
OLD = (
    "        res = aider_runner.aider_heal(pdir, target, error_text, test_command=test_command,\n"
    "                                      read_only_files=_ro_files or None)\n"
    "        return {\n"
    "            \"ok\":         bool(res.ok),\n"
    "            \"target\":     target,\n"
    "            \"error\":      res.error if not res.ok else \"\",\n"
    "            \"duration_s\": res.duration_s,\n"
    "            \"attempts\":   res.attempts,\n"
    "        }"
)
NEW = (
    "        res = aider_runner.aider_heal(pdir, target, error_text, test_command=test_command,\n"
    "                                      read_only_files=_ro_files or None)\n"
    "        heal_ok = bool(res.ok)\n"
    "        heal_error = res.error if not res.ok else \"\"\n"
    "        # S-6 fix: \u043f\u0440\u043e\u0432\u0435\u0440\u044f\u0435\u043c \u0441\u0442\u0430\u0442\u0438\u043a\u0443 \u043f\u043e\u0441\u043b\u0435 heal \u2014 aider \u043c\u043e\u0436\u0435\u0442 \u0432\u0435\u0440\u043d\u0443\u0442\u044c ok=True \u0441 \u0431\u0438\u0442\u044b\u043c \u0444\u0430\u0439\u043b\u043e\u043c\n"
    "        if heal_ok:\n"
    "            try:\n"
    "                _static = static_check(str(pdir / target))\n"
    "                _static_errors = _static.get(\"errors\") or []\n"
    "                if _static_errors:\n"
    "                    heal_ok = False\n"
    "                    heal_error = (\n"
    "                        \"aider heal ok but static errors: \"\n"
    "                        + \"; \".join(e.get(\"message\", str(e)) for e in _static_errors[:3])\n"
    "                    )\n"
    "                    logger.warning(f\"[heal.static] {target}: {heal_error}\")\n"
    "            except Exception:\n"
    "                pass\n"
    "        return {\n"
    "            \"ok\":         heal_ok,\n"
    "            \"target\":     target,\n"
    "            \"error\":      heal_error,\n"
    "            \"duration_s\": res.duration_s,\n"
    "            \"attempts\":   res.attempts,\n"
    "        }"
)
assert OLD in src, "S-6 pattern not found!"
src = src.replace(OLD, NEW)
print("S-6 OK")

# ------------------------------------------------------------------ S-7
OLD = (
    "                new_plan.setdefault(\"build_steps\", plan.get(\"build_steps\", []))\n"
    "                new_plan.setdefault(\"tests\", plan.get(\"tests\", []))\n"
    "                new_plan.setdefault(\"inputs\", plan.get(\"inputs\", []))"
)
NEW = (
    "                # S-7 fix: setdefault \u043d\u0435 \u043f\u0435\u0440\u0435\u0437\u0430\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u0442 \u043f\u0443\u0441\u0442\u044b\u0435 \u0441\u043f\u0438\u0441\u043a\u0438 [] \u043e\u0442 LLM\n"
    "                for _field in (\"build_steps\", \"tests\", \"inputs\"):\n"
    "                    if not new_plan.get(_field):\n"
    "                        new_plan[_field] = plan.get(_field) or []"
)
assert OLD in src, "S-7 pattern not found!"
src = src.replace(OLD, NEW)
print("S-7 OK")

# ------------------------------------------------------------------ validate & write
ast.parse(src)
print(f"\nSyntax OK | {len(src)} bytes | {src.count(chr(10))} lines")
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(src)
print("Written!", TARGET)
print("Now run: git add brain/agents/project.py && git commit -m 'fix(S-1..S-7)' && git push")
