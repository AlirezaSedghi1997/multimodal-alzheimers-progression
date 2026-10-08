# Dependency security and reproducibility

The direct pinned requirements were checked with pip-audit against the configured
advisory service. The recorded output includes known advisories in Pillow and
MONAI; duplicate advisory rows were collapsed by package/advisory ID in the summary.
See dependency_security.json for exact installed and reported fix versions.

No automatic fix or upgrade was applied to the live frozen environment. Changing
libraries while a contract-bound run is active would invalidate its numerical and
software assumptions. This is not a claim that vulnerabilities are harmless.

For review, use the standard-library package checker: it loads neither library.
The package contains no model weights or executable pickle. Never load untrusted
weights, model bundles, images or configs into the historic environment. Keep it
isolated from untrusted inputs and network-serving use. A separate patched fresh
environment, numerical regression and contract version are required before
recommending general-purpose execution. Transitive dependency coverage and exact
reachability still require review; no security PASS is issued for the environment.
