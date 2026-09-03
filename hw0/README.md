# HW0 student handout

Release: Friday, September 4, 2026
Due: Friday, September 18, 2026 at 11:59 p.m. America/Chicago

Complete only the block marked ``Modify your code below'' in each task file. Task 1 asks you to derive `TENNIS_XY`; the matching `TENNIS_UV` pixels are provided. Run each file directly from this folder:

```bash
python task1_homography.py
python task2_projection.py
python task3_epipolar.py
```

Each file has a short, sequential `main()` that shows the full pipeline. It
prints and saves a checkpoint after every subtask, so work from A to B to C and
rerun the same file after each change. Runnable placeholders keep later
checkpoints from crashing while you work. Task 2 uses your Task 1 basketball
result as its background when available.

Student-facing files:

- `task1_homography.py`: editable functions plus the visible 1A--1D pipeline.
- `task2_projection.py`: editable function plus the visible 2A--2C pipeline.
- `task3_epipolar.py`: editable function plus the visible 3A--3C pipeline.
- `hw0_utils.py`: small low-level image, geometry, and plotting utilities; do not edit.
- `report_template.tex`: report template.
- `data/`: all supplied basketball, tennis, mesh, and two-view data.

Use the course computing environment, which provides Python 3.10 or newer,
NumPy, OpenCV, ImageIO, and Matplotlib. Package installation, a custom
command-line interface, a runner, and a test framework are not part of the
assignment.

Submit your revised `task1_homography.py`, `task2_projection.py`, and
`task3_epipolar.py`, the generated `outputs/` folders, and your PDF report.
Submitting outputs or a report without all three revised task files is
incomplete.
