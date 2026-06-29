# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Beyond-accuracy analysis of the instrumented agent traces (data/agent_traces.json):
# (1) tool-usage profile, (2) tool<->cause correlation, (3) investigation depth & stop reason,
# (4) EIG realized information gain, (5) depth->accuracy / agentic-advantage mapping.
import json, collections, statistics as st

tr = json.load(open("data/agent_traces.json"))
n = len(tr)
esc = [o for o in tr if o["n_tools"] > 0]      # cases that escalated beyond the mandatory log
flat = [o for o in tr if o["n_tools"] == 0]    # resolved/decided at the log step

def pct(x, d=n): return f"{100*x/d:.0f}%" if d else "-"

print(f"\n================  AGENT-BEHAVIOUR ANALYSIS  (n={n} cases)  ================\n")

# ── 1. tool-usage profile ────────────────────────────────────────────────────
print("1. TOOL-USAGE PROFILE")
tool_ct = collections.Counter(t for o in tr for t in o["tools"])
print(f"   escalated beyond the log (used >=1 tool): {len(esc)}/{n} ({pct(len(esc))})")
print(f"   resolved at the mandatory log step alone : {len(flat)}/{n} ({pct(len(flat))})")
print(f"   mean escalation tools per case           : {st.mean(o['n_tools'] for o in tr):.2f}")
print("   tool call frequency (escalation tools, deep_log excluded):")
for t, c in tool_ct.most_common():
    print(f"      {c:3}  {t}")
if not tool_ct: print("      (none)")

# ── 2. tool <-> cause correlation ────────────────────────────────────────────
print("\n2. TOOL -> DIAGNOSED FAMILY  (is the investigation targeted?)")
by_tool = collections.defaultdict(collections.Counter)
for o in esc:
    for t in set(o["tools"]):
        by_tool[t][o["pred_bucket"]] += 1
for t, cc in sorted(by_tool.items(), key=lambda x: -sum(x[1].values())):
    tot = sum(cc.values())
    dist = ", ".join(f"{k} {v}" for k, v in cc.most_common())
    print(f"   {t:22} (n={tot}): {dist}")

# ── 3. investigation depth & stop reason ─────────────────────────────────────
print("\n3. INVESTIGATION DEPTH  (effort scales with difficulty?)")
depth = collections.Counter(o["steps"] for o in tr)
for k in sorted(depth):
    print(f"   {k} step(s): {depth[k]:3}  ({pct(depth[k])})")
print(f"   mean steps {st.mean(o['steps'] for o in tr):.2f} | max {max(o['steps'] for o in tr)}")
stop = collections.Counter(o["stop"] for o in tr)
print("   stop reason:", ", ".join(f"{k} {v} ({pct(v)})" for k, v in stop.most_common()))

# ── 4. EIG realized information gain ─────────────────────────────────────────
print("\n4. EXPECTED-INFORMATION-GAIN BEHAVIOUR  (do investigative steps pay off?)")
drops = [o["entropy_start"] - o["entropy_end"] for o in tr
         if o["entropy_start"] is not None and o["entropy_end"] is not None]
esc_drops = [o["entropy_start"] - o["entropy_end"] for o in esc
             if o["entropy_start"] is not None and o["entropy_end"] is not None]
print(f"   mean entropy at start          : {st.mean(o['entropy_start'] for o in tr if o['entropy_start'] is not None):.2f} bits")
print(f"   mean entropy at decision        : {st.mean(o['entropy_end'] for o in tr if o['entropy_end'] is not None):.2f} bits")
print(f"   mean entropy reduction (all)    : {st.mean(drops):.2f} bits")
if esc_drops:
    print(f"   mean entropy reduction (escalated): {st.mean(esc_drops):.2f} bits over {len(esc_drops)} cases")
# realized per-tool gain (information_gain logged per belief update)
tool_gain = collections.defaultdict(list)
for o in tr:
    for sig, g in zip(o.get("signals", []), o.get("info_gains", [])):
        if sig and sig != "deep_log_analysis":
            tool_gain[sig].append(g)
if tool_gain:
    print("   realized information gain per escalation tool (mean bits):")
    for t, gs in sorted(tool_gain.items(), key=lambda x: -st.mean(x[1])):
        print(f"      {t:22} {st.mean(gs):+.2f}  (n={len(gs)})")

# ── 5. depth -> accuracy / agentic-advantage mapping ─────────────────────────
print("\n5. DEPTH -> ACCURACY  (where does the agentic loop help?)")
def acc(group):
    g = [o for o in group if o["correct"] is not None]
    return f"{sum(o['correct'] for o in g)}/{len(g)} ({pct(sum(o['correct'] for o in g), len(g))})" if g else "-"
print(f"   accuracy, resolved-at-log cases : {acc(flat)}")
print(f"   accuracy, escalated cases       : {acc(esc)}")
print(f"   accuracy, overall               : {acc(tr)}")
# off-log signal: of escalated cases, what families were the causes?
print("   escalated-case diagnosed families:",
      ", ".join(f"{k} {v}" for k, v in collections.Counter(o["pred_bucket"] for o in esc).most_common()))
print("   log-resolved diagnosed families  :",
      ", ".join(f"{k} {v}" for k, v in collections.Counter(o["pred_bucket"] for o in flat).most_common()))
print()
