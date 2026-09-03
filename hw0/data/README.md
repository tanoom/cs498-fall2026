# HW0 data

The required assignment uses the State Farm Center basketball scene. The
tennis clips provide a second, fixed-camera application of the same planar
geometry.

## Basketball and epipolar data

- `court/court.png`: State Farm Center scene used for planar and 3D insertion.
- `court/court_top_down.png`: metric court and placement diagram.
- `court/logo.png`: transparent 6 m by 3 m insertion graphic.
- `court/bunny.ply`: mesh used for projection-matrix validation and 3D insertion.
- `epipolar/images/0000.png`, `0005.png`: two-view image pair.
- `epipolar/all_good_matches.npy`: 200 evaluation correspondences.
- `epipolar/eight_good_matches.npy`: eight correspondences used to estimate `F`.

## Tennis transfer data

`tennis/` contains five fixed-camera clips, eight supplied image anchors per
clip, a regulation top-down diagram, checksums, and schema documentation. Use
`match005` for the required transfer and all five clips only if attempting the
bonus. Load the `.npz` files with `allow_pickle=False`. The archives do not
contain metric anchor coordinates, reference homographies, or staff answers.
