# data/

Not tracked in git. `scripts/prepare_rmd17.py` fills it:

```
data/raw/rmd17/                  rmd17_<mol>.npz + official split CSVs (md5-checked)
data/processed/md17/<mol>/
    1k/{train,val,test}/         950 / 50 / 1,000 frames (official split 1), LMDB + nequip_npz.npz
    n100/, n250/                 nested low-data training subsets (val/test link to 1k)
    10k/                         StABlE's reference-h(r) split (symlink to 1k by default)
data/processed/contiguous-md17/  only so StABlE's logged VACF runs; not meaningful on rMD17
```

The on-disk name stays `md17` because StABlE's code branches on it; the labels are rMD17.
