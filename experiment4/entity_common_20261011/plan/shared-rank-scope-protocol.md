# Full TRAIN scope and guard-scorer scaling preflight

Preparation only, no new graph optimizer or utility claim. Use all68334 old
TRAIN head/relation query keys, original TRAIN-selected TRUE guard rows, exact
same common initial DistMult rows (11258x200) and relations (474x200), three
seeds17/29/43. All11258 candidates are scored by the023R3 fixed binary-tree
float64 path. Existing canonical failed records and original evaluation
arithmetic remain unchanged. No VALID or TEST reads during this preflight;
the reused foundation/source was already selected in the exposed development
comparison and cannot become an unseen holdout through this preparation.

Eligibility: protect source-selected old TRUE answers with actual initial
full-pool strict-score rank<=10. For each old TRAIN key, select original7505
pool predicted Top10 by score, ties by lower entity ID; retain a prediction
query only if every selected member remains rank<=10 in the initial11258
pool. Report initial ineligibility explicitly. Compile both full prediction
and scoped prediction contracts. Scoped prediction exempts newTRAIN keys;
TRUE anchors remain included. Keep every compiled reference member, deduplicate
identical references, and group only by the exact same query key and rank
limit. Verify the current minimum reference score on every compiled group.

Measure full-candidate scoring wall time, total preparation time, actual peak
allocated GPU memory and score-matrix bytes. Pin1200s per producer, one GPU
per seed, available GPUs3/4/5 after live peer claims and usage checks; reserve
existing KGEditor processes. No parameter updates or utility selection occurs.

Independent audit: source identities and hashes, every TRUE mapping and
prediction member identity, all affected keys from newTRAIN, exact compiled
full/scoped references, no omitted eligible TRUE member, grouped counts and
all output hashes. Fixed128 randomly selected query rows per seed are rescored
on CPU with an independently written binary tree across every11258 candidate;
scores must match bitwise and prediction/reference fixture ranks exactly.
This is complete structural scope verification plus sampled whole-row score
verification, not independent score reproduction of all68334 rows.

Keep arrays/score fixtures in remote artifacts; promote only lightweight
audited receipts and audits. Neither successful scaling nor smaller grouped
counts establish useful redistribution, algorithmic novelty or goal completion.
