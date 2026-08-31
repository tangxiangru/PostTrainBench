# Commit guards for this checkout

Not installed by cloning. One command per checkout:

```bash
printf '#!/bin/bash\nguard="$(git rev-parse --show-toplevel)/ptb_ops/hooks/pre-commit"\n[ -x "$guard" ] || exit 0\nexec "$guard" "$@"\n' > .git/hooks/pre-commit
chmod 700 .git/hooks/pre-commit
```

## `pre-commit` — do not replace a script a running job is still reading

bash reads a script incrementally. A `git commit` that changes a file writes a **new inode**
at that path, and a job whose shell is part way through the old one gets `ESTALE` the next
time it reads — which is when the long command in front of it returns. The failure therefore
lands hours after the commit and names neither the commit nor the file. That is how 82165
and 82166 lost ten H100-hours each.

`src/commit_utils/pin_src_locally.sh` is the real fix and every current launcher uses it: it
copies the shell half of `src/` to node-local disk, so a job that pins is immune to all of
this. The guard exists for jobs submitted *before* the pin landed — their batch script is
frozen in Slurm's spool and cannot be repaired, so the only protection left is on the
writing side.

It refuses a commit only when **both** hold:

1. a staged path is one a job reads *after* it has started — `src/run_task.sh` and the three
   files it sources or runs mid-flight, plus `src/baselines/run_baseline.sh`; and
2. a PENDING or RUNNING job under this uid **invokes** that entry point and does **not**
   pin.

Everything else commits normally, including every path in this repository that is read once
at job start or not at all.

Three narrowings are load-bearing, each of which was a false refusal first:

- **Comments are stripped before matching.** `ptb_greedy_board.sbatch` names `run_task.sh` in
  a comment explaining what it deliberately does not do. A guard that reads prose refuses
  commits on the strength of prose.
- **Scope is this uid.** Another account's job cannot be reading this checkout and its batch
  script is not ours to read; asking anyway turned five `froilanchoi` autolab jobs into
  "unreadable, so not cleared" vetoes. Several humans share this uid, so this is *not*
  "my jobs" — it is "jobs that could be in here".
- **`scontrol write batch_script` will not overwrite a file that exists.** Handed a
  `mktemp` file it returns 0 and writes nothing, and every `grep` afterwards reads a missing
  path and answers "no match" — which is indistinguishable from "this job is fine". It gets
  a name inside a `mktemp -d` instead, and an empty result is treated as unreadable.

`--no-verify` overrides it for one commit. That is the right answer when you have checked
that every job it names is yours and that losing it is acceptable.
