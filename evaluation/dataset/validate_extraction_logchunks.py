# --- ensure repo root and evaluation/ are importable regardless of CWD ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.isdir(_os.path.join(_r, "src")):
    _r = _os.path.dirname(_r)
_sys.path[:0] = [_r, _os.path.join(_r, "evaluation")]
# -----------------------------------------------------------------------
# Validate our log extractor against the LogChunks benchmark (Brandt et al., MSR'20):
# 816 Travis CI logs, each with a human-labeled, developer-cross-validated "Chunk" that
# describes why the build failed. We run the FORMAT-AGNOSTIC part of our extractor (the
# real-error anchoring) on each raw log and measure CHUNK RECALL: does our extracted window
# capture the human-labeled failure chunk?
#
# Honest scope: LogChunks is Travis CI; our corpus is GitHub Actions. The GHA-specific marker
# logic (##[error]/##[group]) does not apply here, so this measures the GENERALIZABLE
# error-pattern anchoring only.
import os, re, sys, glob, html, collections
import xml.etree.ElementTree as ET
sys.path.insert(0, ".")
from src.apa.agent import _focus_log_on_error

BASE = "data/logchunks/LogChunks"
XML_DIR = os.path.join(BASE, "build-failure-reason")
LOG_DIR = os.path.join(BASE, "logs")

def norm(s):
    return re.sub(r"\s+", " ", html.unescape(s or "")).strip().lower()

def chunk_signature_lines(chunk):
    """The lines most likely to be THE error (longest, most token-rich), used to test recall."""
    lines = [l for l in (chunk or "").splitlines() if len(l.strip()) > 12]
    # drop the '^---- failure generated from ...' provenance pointer LogChunks appends
    lines = [l for l in lines if not l.strip().startswith("^----")]
    lines.sort(key=lambda l: len(l), reverse=True)
    return lines[:3]

def main():
    xmls = glob.glob(os.path.join(XML_DIR, "**", "*.xml"), recursive=True)
    xmls = [x for x in xmls if "__MACOSX" not in x]
    print(f"{len(xmls)} repo annotation files")

    total = recall_hit = strict_first = no_error_anchor = 0
    by_cat = collections.Counter(); by_cat_hit = collections.Counter()
    misses = []
    for xp in xmls:
        try:
            root = ET.parse(xp).getroot()
        except Exception:
            continue
        for ex in root.findall(".//Example"):
            logrel = (ex.findtext("Log") or "").strip()
            chunk = ex.findtext("Chunk") or ""
            cat = (ex.findtext("Category") or "?").strip()
            if not logrel or not chunk.strip():
                continue
            logpath = os.path.join(LOG_DIR, logrel)
            if not os.path.exists(logpath):
                continue
            try:
                log_text = open(logpath, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            total += 1; by_cat[cat] += 1

            window = _focus_log_on_error([log_text], max_chars=2500)
            nwin = norm(window)
            sig = chunk_signature_lines(chunk)
            # RECALL: does our window contain the chunk's most distinctive line?
            hit = any(norm(s) and norm(s) in nwin for s in sig)
            # STRICT: does our window contain the chunk's first substantive line?
            first = next((l for l in chunk.splitlines() if len(l.strip()) > 12 and not l.strip().startswith("^----")), "")
            strict = bool(norm(first)) and norm(first) in nwin
            if hit: recall_hit += 1; by_cat_hit[cat] += 1
            if strict: strict_first += 1
            if not hit and len(misses) < 12:
                misses.append((logrel, sig[0][:80] if sig else "", window[:80].replace("\n"," ")))

    print(f"\n=== EXTRACTION VALIDATION vs LogChunks (n={total}) ===")
    print(f"  CHUNK RECALL (our window captures the labeled failure line): {recall_hit}/{total} = {recall_hit/total:.1%}")
    print(f"  STRICT (captures the chunk's first line):                    {strict_first}/{total} = {strict_first/total:.1%}")
    print(f"\n  recall by structural category:")
    for c in sorted(by_cat, key=lambda c: -by_cat[c]):
        print(f"    cat {c}: {by_cat_hit[c]}/{by_cat[c]} = {by_cat_hit[c]/by_cat[c]:.0%}")
    print("\n  sample misses (labeled line vs our window head):")
    for lr, s, w in misses[:8]:
        print(f"    - {lr.split('/')[-1]}")
        print(f"        labeled: {s}")
        print(f"        ours   : {w}")

if __name__ == "__main__":
    main()
