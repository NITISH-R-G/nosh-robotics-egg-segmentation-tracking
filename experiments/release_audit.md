# Release audit (strict, evidence-based)

Scope: no new methods, no tuning, no edits to frozen configs or evaluation outputs, no evaluation re-run, nothing published. Results are PASS / FAIL / NOT VERIFIED. Machine-generated evidence: `experiments/task1_mask_audit.json|md`, `experiments/video_artifact_audit.json`, `experiments/consistency_checks.json` (32 checks, 0 FAIL), produced by `src/analysis/audit_task1.py`, `audit_video.py`, `audit_consistency.py` (read-only; they need PyMuPDF, installed only in the system Python, not in the lock file).

## 1. Experimental integrity
| check | result | evidence |
|---|---|---|
| Frozen files unchanged and self-consistent (`S1_v1.json`, `classical_v1.json`) | PASS | artifact self-hash recomputed OK; all `source_files_sha256` match current files |
| Frozen S1 evidence hashes match (split manifest, cache audit, candidates.jsonl, protocol) | PASS | `configs/frozen/S1_v1.json` |
| Frozen classical evidence hashes match | **FAIL (explained, not altered)** | `experiments/classical_log.csv` hash differs from `classical_v1.json` (see chronology) |
| `classical_log.csv` differs from the frozen record ONLY by the appended last line | PASS | sha256 of the file minus its last line equals the hash recorded in `classical_v1.json` |
| Historical results edited to make hashes match | PASS (none) | no log or frozen file was rewritten |
| Evaluation ran once per method on exactly 22 eval ids | PASS | `outputs/eval/RUN_RECORD.json` (single record; `final_eval.py` raises if it exists), all 44 eval mask files and the S1 manifest (22 rows) timestamped 13:06 on 2026-11-01, `eval_split.csv` = the 22 eval ids of the split manifest |
| Evaluation re-run during this audit | PASS (not run) | no `final_eval` invocation; `RUN_RECORD.json` unchanged |
| RUN_RECORD artifact hashes = frozen artifact hashes | PASS | `consistency_checks.json` |

**Chronology (2026-11-01, file modification times):** 13:03:23 `final_evaluation_protocol.md` written -> 13:05:51 both frozen files written (hashes of the protocol, split, cache audit and `classical_log.csv` recorded) -> 13:06:25 single evaluation run (`RUN_RECORD.json`, `final_eval_per_image.csv`) -> 13:07:14 eval counts labelled (`data/review/eval_egg_counts.csv`, after predicted counts had been printed, disclosed) -> 13:10:31 `classical_log.csv` gained its `FREEZE-classical_v1` row.
Frozen source state: the 16-line log at freeze time (everything up to and including `DECISION-after-review`). Post-freeze change: one appended administrative line recording the freeze itself (no metric, no config change). The append happened AFTER the evaluation run, so it could not have influenced it. The mismatch is therefore a bookkeeping artefact; the frozen file was deliberately not edited and the log was not reverted. `final_eval.py` was not re-run (a re-run would stop at the run-once guard anyway, and its evidence-hash check would also fail on this one file).

## 2. Task 1 outputs (`outputs/task1_sam_masks/`)
| check | result | evidence |
|---|---|---|
| 110 expected images = 110 mask files = 110 manifest rows | PASS | `experiments/task1_mask_audit.json` |
| Missing / unexpected / unreadable / malformed (non-binary, wrong ndim) masks | PASS (none) | same |
| Mask size equals source image size for every image | PASS | same (110/110) |
| File hash equals manifest and equals the S1 selector output it was copied from | PASS | same |
| Masks altered by this audit | PASS (not touched) | |
| Empty masks | **15** (13 dev, 2 eval) | `not_damaged_` + 10, 11, 12, 13, 20, 42, 52, 116, 117, 122, 123, 124, 125, 135, 145 (eval: 10 and 122) |
| All 13 dev empties carry status `review_no_candidate` (flagged, not silently empty) | PASS | re-derived with the frozen S1 selector on dev candidates (read-only check, scratch) |
| README and report state "15 empty: 13 dev, 2 eval" | PASS | `README.md`, `report/REPORT.pdf` |

## 3. Task 4 video artifacts
| check | result | evidence |
|---|---|---|
| `outputs/video/final/egg_tracking_T1.mp4` opens and decodes completely | PASS | 760/760 frames decoded |
| Properties | 1280x720, 24.0 fps, 760 frames, 31.67 s (source frames 240-999) | `experiments/video_artifact_audit.json` |
| Tracker variant in the video = T1 (`T1_gated_plus_two_egg_prior`, tracker sha `32302c832602`) | PASS | export called with T1 `tracks.csv`; 1256 box-edge pixel checks of the decoded video against T1 CSV boxes, 0 failed |
| 4 Hz update behaviour accurately described | PASS | boxes of sampled frame 240+6k are held for frames 240+6k..245+6k (rule in `src/tracking/export_video.py`, caption in the video says "updated at 4 Hz, held between updates"); the README/results say the same |
| Reported metrics = saved annotations/result files | PASS | `evaluate()` recomputed from both `tracks.csv` files for dev/held-out/all matches `experiments/video_tracking_log.csv` and `outputs/video/evaluation.json` (coverage, misses, extras, ID switches, distinct IDs) |
| Video uploaded / Google Drive link exists | **NOT DONE** | no link; report and README say so |
| The video file is not in the repository file list | PASS | `*.mp4` ignored |
Not changed: tracker and detector were neither improved nor re-run.

## 4. Environment and reproducibility
| check | result | evidence |
|---|---|---|
| `reportlab==4.4.10` and its dependency `charset-normalizer==3.4.4` added to `requirements.lock.txt`; the PyTorch CUDA wheel index line `--extra-index-url https://download.pytorch.org/whl/cu128` added (needed for `torch==2.11.0+cu128`) | PASS | no other version changed |
| Fresh Python 3.12.3 venv (`scratch/envtest/venv`), `pip install -r requirements.lock.txt` | PASS | exit 0; `pip freeze` equals the lock file exactly, nothing extra installed (wheels came partly from this machine's pip cache) |
| Full test suite in that fresh venv | PASS | 197 passed |
| `src/report/build_report.py` in that fresh venv | PASS | `report/REPORT.pdf` rebuilt (includes the corrections below) |
| Test suite in the main `.venv` | PASS | 197 passed |
| README commands for data download, SAM checkpoint download, yt-dlp video download | NOT VERIFIED | not re-executed; the Roboflow zip, Meta checkpoint and YouTube video must be fetched by the user |
| Full pipeline from raw data (SAM cache ~30 min, classical experiments, video detection) in a fresh clone | NOT VERIFIED | not performed; tests and the report build were run in the working tree, not a fresh clone |
| `src.data.splitting` regenerates `data/split_manifest.csv` | NOT VERIFIED | README wording softened accordingly |
| GPU-dependent SAM step in a fresh environment | NOT VERIFIED | needs the cu128 wheel on an NVIDIA GPU; installation only was tested |
| `yt-dlp` / `yt-dlp-ejs` installation | NOT VERIFIED | not in the lock file; versions noted in README from the tools environment |

## 5. Cross-artifact consistency (`consistency_checks.json`: 32 checks, 0 FAIL)
Verified PASS: split counts 88/22/110; eval ids; frozen classical config = `configs/classical_final_dev.json` (sha `6696e8cab3ba`); dev count metrics (S1 73/88, classical 42/88; S1 over 1/under 14); dev classical instance P/R/F1 0.442/0.393/0.416 (TP/FP/FN 42/53/65), macro IoU/Dice 0.363/0.441 over 81 non-empty-reference images; eval counts (S1 19/22, classical 12/22; totals 22/26 vs 25 expected); eval agreement (IoU 0.315, Dice 0.439 over 20; instances 6/20/16); 14,400 candidates and 16.4 s/image; SAM checkpoint sha equals run record and the file on disk; torch `2.11.0+cu128` / Python 3.12.3 equal the frozen environment record; the report contains the hashes and headline numbers; every directory-qualified path named in the README, results documents and report exists.

**Discrepancies found and corrected after verification (transparent corrections):**
1. `README.md` said S1 "has no false positives on dev"; the data show one over-count in 88 (verdict precision 0.990). Corrected.
2. `experiments/final_evaluation_analysis.md` said "no false positives" for S1 on eval; the evidence is count-based, so it now says "no over-counts".
3. "Seven unresolved images" (report, `classical_failure_analysis.md`) were in fact the 7 images whose bad-boundary S1 candidate was excluded from the reviewed reference (6 confirmed, 1 uncertain; `data/review/dev_manual_selection_manifest.csv`). Reworded; the number was right, the label was misleading.
4. `final_evaluation_analysis.md` called eval images 10 and 122 "known class-B failures"; that was not checked on eval images. Reworded to a stated resemblance.
5. `final_evaluation_analysis.md` named a non-existent path `sheet_0..5.jpg`; replaced by "sheet_0.jpg to sheet_5.jpg".
6. The report had been built with the system Python 3.14, not the documented Python 3.12 environment; it was rebuilt in the clean 3.12 venv.
7. README setup now states Python 3.12 (`py -3.12`), the lock file's CUDA index, `reportlab` and the yt-dlp versions; the "re-running reproduces the split" claim was softened.
Not re-verified (inherited from earlier sessions, kept as stated): selector-experiment and boundary-refinement numbers (`experiments/selector_results.md`, `boundary_refinement_results.md`), the verdict precision/recall 0.990/0.876, the 20 right-count-wrong-instance images, per-scene F1 values and the "S1 selector 0.26 s" figure; they are quoted from saved files but were not recomputed here.
AI-assisted annotation disclosure present and consistent in: README, report (honesty statement and section 10), `final_evaluation_results.md`, `data/review/eval_egg_counts.csv`, `video_tracking_protocol.md`. PASS.

## 6. Public repository preparation
| check | result | evidence |
|---|---|---|
| Secrets / API keys / tokens / emails in proposed public files | PASS (none found) | regex scan of the 376 proposed files (text) and the report text; the proposed list was produced with `git ls-files -o --exclude-standard` against a throwaway bare repo in `scratch/` (no repository was created in the project folder) |
| Local absolute paths or the user name in proposed files | PASS (none found) | same scan; PDF text scan found none |
| Personal data in images | NOT VERIFIED | image/PDF content and EXIF were not inspected exhaustively; dataset photos show hands (dataset licence, no identities known) |
| `.venv-tools/` (yt-dlp tool environment) was not ignored | FIXED | now in `.gitignore`; it contained local paths |
| Unnecessary caches / virtualenvs / temp files excluded | PASS | `.venv/`, `.venv-tools/`, `scratch/`, `__pycache__/`, `.pytest_cache/`, `.claude/`, `outputs/*` (allow-list below) |
| Files too large for GitHub | PASS | proposed set about 12 MB; largest file `report/REPORT.pdf` 2.3 MB (limit 100 MB per file); excluded: SAM checkpoint (about 375 MB), dataset, video files |
| Duplicate artifacts | NOTE | the 22 eval S1 masks exist twice (`outputs/eval/S1/masks` and `outputs/task1_sam_masks/masks`) by design |
| Dataset licence / CC BY 4.0 attribution | ADDED to README | Roboflow user fire-bf7qe, link, licence; overlays contain dataset images |
| Third-party video content | RISK, decision needed | Mocomi YouTube video; small stills are in `outputs/video/examples/` and `outputs/video/contact_sheet.jpg`; the annotated MP4 reproduces the whole experiment interval and is intended for a Drive link only |
| Code licence | MISSING | none chosen; needs the author's decision |
| Referenced files exist / setup instructions present | PASS / PASS | consistency check; README Setup + Run |

### Proposed public repository contents (376 files; full list `experiments/proposed_public_files.txt`)
`.gitignore`, `README.md`, `requirements.lock.txt`, `report/REPORT.pdf`; `src/**` (data, segmentation, evaluation, experiments, analysis, tracking, report); `tests/**` (24 files); `configs/` (`classical_final_dev.json`, `frozen/`, `withdrawn/`); `data/` (split manifests, grouping decisions, AI-assisted review labels, `data/video/`); `experiments/**` (protocols, logs, generated results, audits, this file); `outputs/figures/`, `outputs/task1_sam_masks/` (110 masks + manifest), `outputs/eval/` (S1/classical masks, overlays, RUN_RECORD, eval_split), `outputs/classical/final_overlays/`, `outputs/video/examples/`, `outputs/video/evaluation.json`, `outputs/video/contact_sheet.jpg`.

### Keep local or distribute separately
- Roboflow dataset (`data_raw/`, `data_raw_dl/egg.zip`) -> download from the source link.
- `models/sam_vit_b_01ec64.pth` -> Meta download.
- Source video (`scratch/video_dl/*`) -> re-download with yt-dlp.
- `outputs/video/final/egg_tracking_T1.mp4` (11.7 MB) -> Google Drive (after your approval).
- SAM candidate cache `outputs/sam_candidates/` (masks, candidates.jsonl; regenerable, needed to re-run S1 selection), other generated `outputs/*` working folders, `scratch/`, `.venv*`, `.claude/`.

## 7. Release checklist
- [x] Frozen configs intact; chronology and the log append documented
- [x] Task 1: 110 masks verified; 15 empty IDs listed; docs match
- [x] Video artifact verified (readable, 760 frames, T1, 4 Hz behaviour, metrics reproduce)
- [x] `reportlab` locked; fresh 3.12 venv install, 197 tests and report build pass
- [x] README: setup, run, attribution, AI disclosure; report PDF rebuilt
- [ ] Decide code licence; decide whether the video stills may stay in the repo
- [ ] Your approval to run `git init`/commit (a local commit has NOT been made; there is no repository yet)
- [ ] Your approval to create the public GitHub repository and push
- [ ] Your approval to upload the annotated MP4 to Google Drive; set sharing to public; paste the verified link into the report (rebuild) and README; open the link logged-out to verify
- [ ] Human spot-check of AI-assisted labels if time allows (not required, but the report states they are unverified)

## 8. Unresolved blockers for public submission
1. No Drive link (required by the assignment inside the PDF). 2. No GitHub repository. 3. Licence and third-party-video decisions. 4. NOT VERIFIED items in sections 4 and 6 (fresh-clone full pipeline, downloads, personal data in images).
Ready for publication? **Not yet**: the content is internally consistent and reproducible for tests and report generation, but the two required public artifacts (repo, Drive link) do not exist and need your explicit approval to create.

## 9. Publication preparation (local only; nothing published, no remote, no upload)
Decisions applied: MIT `LICENSE` for original code (copyright line "NITISH-R-G", the git user name; change it if you prefer your legal name); dataset CC BY 4.0 attribution kept in README; MP4 excluded from Git (`*.mp4`, `outputs/video/final/`); video stills reduced to the two the report embeds to explain the identity-swap failure (`outputs/video/examples/failure_P_carried_identity_swap_726_759.jpg`, `T1_dense_672_810_P_picked_up_swap.jpg`); `contact_sheet.jpg`, `dense_*.jpg` sheets and the other examples are ignored (regenerable; README says so). `tracks.csv` of both variants, `frame_stats.csv` and `evaluation.json` are tracked (no imagery) so the video results can be re-checked. `.gitattributes` (`* -text`) disables line-ending conversion so committed bytes equal the working-tree bytes that the frozen hashes were computed on (checked for the split manifest, protocol, frozen config, report and a source file: identical).
Local repository: `git init -b main`; repo-local author `NITISH-R-G <NITISH-R-G@users.noreply.github.com>` (the globally configured personal email was NOT used, so it will not appear in the public history; the commit is local and can be re-authored before any push).

**Privacy and licensing findings (what was actually inspected):**
- Text files (all tracked `.py/.md/.csv/.json/.txt`): regex scan for local paths, user name, emails, key/secret/password/token strings: only benign hits (pytest `@pytest.fixture`, the word "tokenizers", the LICENSE holder name). PASS.
- 196 images (PNG/JPG): EXIF present in 0; the only embedded text metadata is Matplotlib's "Software" tag in 3 figures. PASS.
- `report/REPORT.pdf`: metadata author "(anonymous)", producer ReportLab, no URLs and no local paths in the extracted text. PASS.
- Visual content: the 22 eval overlays (6 sheets) and the Task 3 figures were viewed in this project; the classical overlay sheets were viewed in an earlier session and not re-checked for this review; the binary mask and instance PNGs and the video stills were not each viewed. The dataset photos contain hands/skin; no faces or identifying text were noticed in the viewed images. This is NOT an exhaustive visual privacy review.
- Dataset licence: CC BY 4.0 attribution (Roboflow Universe user fire-bf7qe, link, changes noted) is in the README; the images themselves are not tracked, but overlays contain individual dataset images under that licence. The dataset's `README.dataset.txt`/`README.roboflow.txt` are not copied into the repo (they sit in the ignored `data_raw/`); the README carries the required attribution instead.
- The MIT licence covers original code only (stated in `LICENSE` and the README); SAM is Apache-2.0 (not redistributed); the YouTube/Mocomi video is third-party and not tracked, except the two small stills above (residual risk, flagged to the author).

## 10. Local verification record (before the initial commit)
- Full test suite (main `.venv`, Python 3.12.3): 197 passed. README `-m` modules: 13/13 importable; README backticked paths: 0 missing (excluding data/weights/video paths that are documented as external or regenerated). Report: opens, 8 pages, 10,743 text characters.
- Tracked: 377 files, 9.75 MB of content (before compression); no `.mp4/.pth/.zip/.npz`, no `.venv*`, `scratch/`, `models/`, `data_raw/` paths tracked. Largest tracked file `report/REPORT.pdf` (2.3 MB).
- Ignored (not tracked): `.venv`, `.venv-tools`, `.pytest_cache`, `data_raw*`, `models`, `scratch` (including the 4.9 GB test venv), `outputs/sam_candidates`, `outputs/selector_dev`, `outputs/classical/E00-E13,final`, other working output folders, the MP4 and non-essential video stills.
- Initial commit `6b6ed81659540614d9a0aa0ab7eaee4b6167e766` on branch `main`: "Egg segmentation and tracking: CV intern task submission (local)". This file was committed in it without the hash (a commit cannot contain its own hash); the hash is recorded by the follow-up commit that adds this section. No remote is configured; nothing was pushed.

## 11. Pre-publication changes (supersede the stills, author and video statements in sections 3, 6, 9 and 10)
- **Third-party stills removed from the project's public history.** The two stills (`failure_P_carried_identity_swap_726_759.jpg`, `T1_dense_672_810_P_picked_up_swap.jpg`) were in the first local commit. The repository history was rebuilt as a fresh root (orphan branch) so no commit or object contains them; the original local history (commits `6b6ed81`, `efb7a51`) and the two files are kept only in the git-ignored `scratch/evidence_third_party/` (a `git bundle` plus copies), never to be pushed. The old branch was deleted and unreachable objects pruned; absence is verified by hashing the two files and searching every object of every ref (see section 12). Source of the permission concern: the video (MocomiKids, YouTube id Y4JyV94y2_A) carries the standard YouTube licence (`license=NA` in the YouTube metadata); no permission statement was found, and a web search found no Mocomi licensing page. Permission has NOT been requested or obtained (contacting the rights holder is the author's decision).
- **Replacement:** `outputs/figures/video_tracking_trajectories.png` (generated by `src/tracking/trajectory_plot.py` from `tracks.csv` only; no tracking result changed) now appears in the report and results text. `contact_sheet.jpg`, `dense_*.jpg` sheets and `outputs/video/examples/` remain untracked and ignored.
- **Videos:** the footage-based annotated MP4 stays local and is not to be published. A footage-free demonstration video was generated from `tracks.csv` only (`src/tracking/schematic_video.py` -> `outputs/video/final/egg_tracking_T1_schematic.mp4`, 760 frames, 24 fps, 1280x720, blank schematic table with boxes, IDs, frame numbers, a growing trajectory plot and the proxy identity-switch annotation). It is not uploaded; both videos are ignored by `*.mp4`. Its content has been inspected on sampled frames only (frames 520 and 810 of the file).
- **Author:** all commits use `NITISH-R-G <225521762+NITISH-R-G@users.noreply.github.com>` (account id 225521762 read from the authenticated GitHub account; the personal email is not used). Whether GitHub links the commits to the account is confirmed only after a push.
- **Report:** rebuilt in the fresh Python 3.12.3 venv from the lock file (`scratch/envtest/venv`); the PDF no longer embeds any video frame (one embedded figure on the tracking page: the trajectory plot).

## 12. Rebuilt history and verification (pre-publication)
- Root commit `d614593e36acf5f1334399ef9783cd1803ed4c40` ("Egg segmentation and tracking: CV intern task submission"), branch `main`, author and committer `NITISH-R-G <225521762+NITISH-R-G@users.noreply.github.com>`; 378 tracked files; this section was added by a second commit (a commit cannot contain its own hash). No remote configured; nothing pushed.
- The previous local history (`6b6ed81`, `efb7a51`) was deleted from the repository, reflog expired and objects pruned (`git gc --prune=now`). Verified: `git cat-file -e` returns "not found" for both still-image blob hashes (`ceb6b788...`, `80b21123...`) and for `efb7a51`; no object path matches `outputs/video/examples`, `contact_sheet` or `dense_`; `git log --all -- outputs/video/examples` is empty; `git fsck` reports nothing unreachable. The only copies are in the ignored `scratch/evidence_third_party/` (bundle of the old history plus the two files).
- Tests after the changes: 197 passed (main `.venv`).

## 13. Final publication gate: schematic video inspection and visualisation corrections
Only `src/tracking/schematic_video.py` and `src/tracking/trajectory_plot.py` changed (visualisation); tracks, tracker, detector, metrics and frozen files are untouched.
- **Clarification of "frame 810":** in section 11 I wrote "frames 520 and 810 of the file". That was wrong. Those were file indices 520 and 570 (0-based); the burned-in caption on index 570 reads "source frame 810" because the caption prints source-video frame numbers (file index + 240). The exported file has 760 frames (indices 0-759 = source frames 240-999). Nothing refers to a file frame 810.
- **Plot lag (expected, not a defect):** the boxes and trajectory only change at the 4 Hz sampled source frames (240, 246, ...) and are held for 6 frames, while the black cursor moves every frame, so the drawn line can trail the cursor by up to 5 frames, and a newly started ID shows a box before its line has a second point. Independently, the cursor position was checked on all 760 decoded frames against the expected pixel column: maximum error 0 px.
- **Defects found and fixed in the visualisation (not in any result):** (1) straight lines were drawn across frames where an ID had no detections (e.g. ID 1 across the hand carry, ID 2 across the pick-up), which implied observed continuity; gaps are now dashed and labelled "interpolation"; the same change was made in `outputs/figures/video_tracking_trajectories.png` (used in the report, which was rebuilt in the clean Python 3.12 venv); (2) boxes were drawn with non-uniform scaling (0.625 horizontal, 0.306 vertical) which distorted shapes; now uniform 0.625 with a vertical crop to the egg band (source y 365-917; all tracked boxes lie in y 386-904); (3) the switch caption started at 744 although ID 2 first appears on the wrong egg at sample 750; events are now "hand picks up P next to Q" (744-749) and "SWITCH ..." (750-815); (4) a "no egg detected at this 4 Hz sample" notice is shown when a sample has no detection; (5) the plot caption now explains solid/dashed lines, the 4 Hz hold and the cursor.
- **Inspection of the final file** (`outputs/video/final/egg_tracking_T1_schematic.mp4`): all 760 frames decode; 1280x720, 24 fps, 31.67 s; frame counter caption consistent with the cursor (0 px error on 760/760 frames); a 19-tile contact sheet over the whole video and full-resolution frames at the start (source 240), hand lift (504), switch (753) and end (999) were viewed: ID labels, boxes and trajectories are readable; the switch annotation appears at the right time; the only cosmetic flaw seen is that the "ID 1" label is partly clipped at the top edge of the table in a few frames, and faint compression ghosting of old labels. Frames between samples were not each viewed. SHA-256 `6889cc0aa24e6f4b84660cd7b82a15782d0aff2364d7fd72d4c445c09fbc462d`, 5,159,150 bytes. Not uploaded.

## 14. T3b adoption (local commit; nothing published after this change)
- **Selected Task 4 tracker: T3b** (decision recorded after the improvement round; see `experiments/tracking_improvement_audit.md` and `tracking_improvement_results.md`). T0, T1, T2, T3 sources, logs and per-frame outputs are preserved unchanged; the improvement log is append-only and includes the sensitivity and failed-criterion runs.
- **Disclosures carried into README, report and results:** 0 identity switches and 3 of 3 retained annotated gaps on this single clip; single-annotator, AI-assisted proxy annotation; T3b designed after inspecting the critical sequence; the held-out block was viewed; not independent generalisation validation; 4 associations flagged uncertain (frames 540, 684, 750, 810, all "ambiguous association"); ID 2 unobserved from frame 762 to 804 (T3b displays a predicted box at 762, 768, 774 only).
- **Correction:** a draft of the improvement audit said the frame-810 association was a two-egg-prior relink; the CSV shows an ordinary "ambiguous association" and the text was corrected before the commit. The previous GitHub-hosted README and PDF (commit `944b904` and earlier) describe T1 as the tracker and say the repository does not exist / is "not published yet"; those statements are superseded by this commit's documents.
- **Proposed demonstration video (not uploaded):** `outputs/video/final/egg_tracking_T3b_schematic.mp4`, SHA-256 `49d7f5645ad8c0dc329661c4894d96847626678e86462a90a3aaca8818910c19`, 5,327,003 bytes; all 760 frames decode, 1280x720 only, 24 fps, 31.67 s; the burned-in current-frame cursor matches its expected column on all 760 frames (max error 0 px; `experiments/t3b_video_artifact_audit.json`). Footage-free; neither the original footage nor the annotated original-footage video is in any public artifact.
- **Verification:** report rebuilt in the fresh Python 3.12.3 venv from the lock file; full suite 213 passed; README/PDF consistency checks in `experiments/consistency_checks.json` (and the T3b-specific string checks recorded in the commit message summary).
