# Author request draft — not sent

Prepared 2026-11-26. The [repository](https://github.com/kyonofx/MDsim) gives
`xiangfu AT mit.edu` as its contact. Xiang Fu's
[current personal site](https://xiangfu.co/) lists `dainves1@gmail.com`.
No email or GitHub issue has been sent or posted.

Subject: Checkpoint and run details for GemNet-T / naphthalene stability result

Hi Xiang,

We are working on a class project using *Forces are not Enough*. We obtained the
PDF attached to the October 29, 2024 EIC revision (edit `yveYNGOUOK`). Table 3
still reports GemNet-T on naphthalene at 1.5 meV/Å force MAE and 8 ± 2 ps
stability. We would like to reproduce that baseline before testing a modified loss.

Could you please help clarify:

1. Was the retained GemNet-T/naphthalene result rerun with the periodic-coordinate
   fix, or was this system unaffected? Which source revision produced that table entry?
2. Is the checkpoint used for that result available, ideally with its
   resolved config, GemNet scale factors, source commit and environment versions?
3. What were its actual training epoch/update count, seed and checkpoint selection
   rule? The released MD17 config caps training at 10,000 epochs, whereas arXiv
   Table 7 says 2,000. We also noticed the released `update_best` condition can
   overwrite `best_checkpoint.pt` when force MAE worsens; was that code used for
   the published model, or was the checkpoint selected separately?
4. The revision's supplementary GemNet YAML omits `loss_energy`, which defaults
   to MSE in its trainer; current GitHub explicitly sets MAE. The supplement also
   lacks the wrapping fix and references a scale JSON absent from the ZIP. Which
   training loss and scale factors were used for the reported naphthalene model?
5. Could you share the five MD starting frame IDs and velocity seeds/momenta,
   individual failure times, and confirmation that the released 500 K,
   0.5 fs Nose–Hoover (`ttime=20`) settings and >0.5 Å bond criterion were used?

For context, we ran only a one-epoch original-MD17 pilot with the corrected
repository and observed no bond failure in a single 20 ps test. We understand
this is not a matched checkpoint or a completed reproduction. We want to resolve
the exact run provenance before funding further training or making comparisons.

Thanks for your time and for releasing the benchmark!
