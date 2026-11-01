# Task 4 video tracking: scope, protocol and honest limits

## Video integrity (checked)
`scratch/video_dl/egg_video.f399.mp4`: AV1, 1920x1080, 24 fps, 1440 frames (60 s), opens with OpenCV, SHA-256 `841d50d85d5f950e306dd2a4049a09c98d1afe1261ef73d3056c049baabc6529`.
`egg_video.f251.webm` is the audio stream only (not decodable as video, unused).

## Experiment interval (observed from the video, `outputs/video/contact_sheet.jpg`, `outputs/video/frame_stats.csv`)
- 0-~224: green title cards ("presents", "Solid Egg vs Liquid Egg"); excluded.
- 224-1004: the table scene with two white eggs, a hand and a checked shirt; the first frames carry overlay captions ("Plane surface", "Raw Egg", "Hard-Boiled egg").
- 1004-1440: dark transition, explanation text card, end card; excluded.
- Analysed interval: frames 240-999 (`src/tracking/sampling.py`). Frames 990-999 are a fade-out / hand entering and are marked ambiguous.

## Tracking objective
Keep two distinct egg identities (P and Q) across the interval despite spinning, motion blur, hand occlusion and temporary detection loss.
Identity names are ours: Q = the egg captioned "Raw Egg" at frame ~336 (left at the start), P = the other egg.
Observed storyline (single annotator, from viewing every sampled frame and a 3-frame zoom of 726-759): the hand lifts Q, carries it to the right and spins it (510-648); carries it back to the far left (648-744);
then picks up **P** and carries/spins it on the right while Q stays at the far left (744-1000). The left/right order of the two eggs therefore changes twice; position order is not an identity cue.

## Detector (`src/tracking/detect.py`, config sha printed in the log)
Fixed rule at 960 px width: grey >= 200 and HSV saturation <= 60, 5 px open, 7 px close, then area 0.2-5% of frame, solidity >= 0.90, aspect <= 1.7, area/ellipse-area >= 0.85.
All values were fixed before the first run on the video. The earlier frozen-candidate classical configuration was tried on 7 development frames first and rejected for this video (it also segments wall, shirt and hand; `outputs/video/explore_classical.jpg`); that is why a video-specific detector exists. It was never fitted to the image-set evaluation split.
Detector parameters were NOT changed after seeing tracker results.

## Tracker variants (declared before the runs; both are reported)
- `T0_gated`: constant-position gated Hungarian association (cost = 0.5 normalised centroid distance + 0.5 (1-IoU)), gate = (1.2 + 0.35*missed_steps) mean-box-diameters, track closed after 24 missed steps. New IDs after closure.
- `T1_gated_plus_two_egg_prior`: T0 plus a scene prior: when two tracks are already open, an unmatched detection is linked to the nearest lost track at any distance instead of starting a third ID. This encodes "there are two eggs" and is an assumption, not a measurement.
No appearance model: the two eggs are visually identical, so no appearance cue can separate them.

## Sample and split
Every 6th frame (4 per second): 127 frames. Development = even 1-second blocks (64 frames), held-out = odd blocks (63 frames). The tracker runs over the whole sequence (time continuity needs both); metrics are reported per subset.
**Disclosure:** the identity annotation (below) was written after viewing overlays of ALL 127 sampled frames, so the held-out frames were seen by the annotator. They were not used to choose any parameter; neither variant was changed after the first run. They are "not tuned on", not "unseen".

## Proxy ground truth and metrics (`src/tracking/evaluate.py`, `data/video/identity_segments.csv`)
Hand-written time segments give, per segment, the number of clearly visible eggs and a rule mapping each detection to P or Q (left/right order or x position). Frames 480, 504, 744, 810 (egg half under a hand) and >= 990 are excluded as ambiguous.
This is single-annotator, AI-assisted, not independent human ground truth. Counted separately:
- **Detection coverage**, **missed eggs**, **extra detections** (segmentation errors; a missed detection is not an identity switch).
- **ID switches**: change of track ID between consecutive detections of the same physical egg. **Extra IDs**: distinct IDs beyond the two eggs (fragmentation).
- **Continuity through occlusion**: for each hand-occlusion gap, whether the egg has the same ID before and after.
- Runtime: detector median ms/frame at 960 px width (CPU), tracker ms/frame. Video decoding (AV1 1080p) is much slower and not included.
