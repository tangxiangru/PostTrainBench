# Node environment acceptance

The Slurm launcher has an explicit `--environment-acceptance humaneval` branch.
It requires held submission and15min walltime, preserves frozen requested nodes,
and exits before provider probing or `run_task.sh`. AWM supplies an independently
registered operation receipt and the frozen source/image specification.

The runner executes fixed invented programs in both real image layouts with
`--nv`. It checks one allocated H100, frozen public data and sandbox dependency
bytes, native verification and durable raw publication, plus outer interruption.
It does not run a scientist or real model and produces no scientific result.

The optional `on_admission` callback is carried across the trusted supervisor's
fresh Python exec by a dedicated pipe. The event follows bootstrap attestation,
is bound to supervisor PID and submitted code hash, and is not exposed to the
untrusted command. Default callers create no admission pipe and retain the
existing execution behavior. The outer probe uses this actual event to arm its
deadline; a parent-process monkeypatch cannot supply this evidence.

The outer test shortens escalation grace to5s and requires termination before
the separate30s inner limit. It checks live exact handles at signalling and
terminal handles afterward. Observation gaps or emergency cleanup cannot pass.
All known handles belong to the created producer tree; no global/name-based
process reclamation is performed.

The persistent report is `acceptance.json` under the frozen output root/cell/job.
Native artifacts stay under `official_eval/` there, including failures. Public
runtime copies stay in scratch and are not archived as experiment evidence.
Consumers must verify the receipt, manifests, actual raw bytes and native case
semantics, not just the top-level status. Node success is narrower than model or
benchmark validation, and only admits the observed node under the frozen runtime.
