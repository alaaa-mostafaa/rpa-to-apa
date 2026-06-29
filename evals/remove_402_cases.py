import json
from pathlib import Path

ACTIVE_PATH = Path("data/benchmark_1000_eig_vs_rpa.json")

repos_to_remove = {
    'aflplusplus/aflplusplus', 'allendang/cimgui-go', 'amzn/ion-js', 'ansible/galaxy',
    'auula/tunadb', 'bavix/laravel-wallet', 'box/box-windows-sdk-v2', 'dart-lang/jnigen',
    'esri/arcgis-runtime-samples-qt', 'google/gts', 'ilestis/miscellany',
    'iotaledger/wallet.rs', 'jdbi/jdbi', 'jgoguen/calibre-kobo-driver',
    'kaspanet/rusty-kaspa', 'kubenetworks/kubevpn', 'lesuisse/vue-dompurify-html',
    'libre-tube/libretube', 'liferay/liferay-blade-samples', 'looker-open-source/sdk-codegen',
    'mainmatter/ember-simple-auth', 'misp/misp-modules',
    'open-telemetry/opentelemetry-dotnet-instrumentation', 'pmeier/pystiche',
    'rcore-os/zcore', 'ritchie46/anastruct', 'rootstrap/rails_api_base', 'servo/html5ever',
    'spencekonde/megatinycore', 'steam-headless/docker-steam-headless', 'tenzir/vast',
    'viaversion/viafabric', 'wordpress/gutenberg'
}

data = json.loads(ACTIVE_PATH.read_text())
before = len(data)
data = [r for r in data if r.get("repo") not in repos_to_remove]
after = len(data)
print(f"Removed {before - after} cases ({before} -> {after})")

ACTIVE_PATH.write_text(json.dumps(data, indent=2))

scorable = [
    r for r in data
    if r.get("rpa", {}).get("judge", {}).get("verdict") not in ("NOT_SCORABLE", "EVALUATION_ERROR", "")
    and r.get("apa_eig", {}).get("judge", {}).get("verdict") not in ("NOT_SCORABLE", "EVALUATION_ERROR", "")
]
apa_correct = sum(1 for r in scorable if r["apa_eig"]["judge"]["verdict"] == "CORRECT")
rpa_correct = sum(1 for r in scorable if r["rpa"]["judge"]["verdict"] == "CORRECT")

print(f"Total remaining: {len(data)}")
print(f"Scorable: {len(scorable)}")
if scorable:
    print(f"APA accuracy: {apa_correct/len(scorable)*100:.1f}% ({apa_correct}/{len(scorable)})")
    print(f"RPA accuracy: {rpa_correct/len(scorable)*100:.1f}% ({rpa_correct}/{len(scorable)})")
