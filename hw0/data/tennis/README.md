# Tennis transfer data

Each `matchNNN.npz` contains only NumPy arrays and scalar values and can be
loaded with `numpy.load(path, allow_pickle=False)`. Its fields are:

- `court_anchor_image_xy`: eight supplied A--H image points in the same order
  used by Task 1;
- `court_anchor_name`: the A--H labels for those image points;
- `image_size_wh`, `valid_frame_range`, `fixed_camera`, and `reference_frame`;
- `court_dimensions_m`: `[23.77, 10.97, 8.23]` for length, doubles width, and
  singles width.

You derive the metric A--H coordinates once in Task 1 and may reuse those
coordinates with each clip's supplied image anchors for the optional bonus.
The archives deliberately omit metric answers, reference homographies, and
private evaluation labels.

`match005` is the required transfer example. The remaining clips exercise
different court surfaces, viewpoints, and frame rates for the optional bonus.

`tennis_court_imperial_wikipedia.png` is the court-dimensions diagram shown in
the handout. It is the Wikipedia/Wikimedia Commons figure
[Tennis court imperial](https://commons.wikimedia.org/wiki/File:Tennis_court_imperial.svg),
created by NielsF and subsequently edited by other Commons contributors. The
file is available under CC BY-SA 3.0 (among the licenses listed on its Commons
page). The handout rotates the diagram by 90 degrees and identifies that
adaptation in the caption.

## SHA-256

```text
abf9b9f5699bcf2ad2770195b13269bff8d61977ba9ae4a2384401c3eb26aac6  match001.mp4
93b4e3ea6a7843325adfd00c1cba2c6f2bab41c23adc57c4b24a072050d4697f  match001.npz
1ebc5c6613fa1be1f9b9ad9ef64a2c6d2c7e9997d104b37422192b9d018795b7  match005.mp4
17bb8f5d7055db8752be92222a8974eec0ee5f19f1d9a63a8aff80ca89e5f353  match005.npz
1e2c6d2df03fbeb2550b8fb5f8d7a97a1b199b4395904813452af159aefce4bb  match028.mp4
d92d52ac58378b728097220a528aeb0887b44df9b2822c0a80747217a0146db7  match028.npz
8c2e4ae64f36ad14f44c27f1a19c8de5224c46ec49406e106d6b99f6a2aac6e6  match091.mp4
548d6cc0320c662f68dbfc8f820741d867b8f4fc03c0e644dc24cc861d9703ac  match091.npz
40bb16df75f24293d7c5672f132587c571163485d9c2b2c7bfac60c7db665052  match125.mp4
7b0564010040d15216d48aee8891540b7c18ed5540a5db5eb8ed2f3d0901a55f  match125.npz
7cedead2a51bf28feb3d4e84974fedf22b7f959781b50af2b0dc3d46e92bd80d  tennis_court_imperial_wikipedia.png
```
