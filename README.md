# PC-FMCW Laser Headlamp Research Project

Fully Python implementation for:

- **Part A:** reproduction-style simulation of the PC-FMCW laser headlamp paper.
- **Part B:** predictive adaptive illumination using VRU trajectory prediction.

Run:

```bash
pip install -r requirements.txt
python run_all.py
```

Outputs are saved in:

```text
figures/
outputs/
```

## Dataset note

By default Part B runs on a lightweight synthetic VRU dataset so the project is reproducible without large downloads.

For Waymo Open Motion Dataset (WOMD), place local `.tfrecord` scenario files in:

```text
data/womd/
```

and run:

```bash
python run_part_b.py --data_source womd --womd_dir data/womd
```

The WOMD loader is lightweight and best-effort. Real WOMD evaluation requires locally downloaded official scenario files.
