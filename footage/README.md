# ODC footage staging

Local-only. NEVER commit video files — they stay on the demo machine.
Only this manifest (URLs + licenses) is tracked.

## Rules
- Every clip: free license (Pexels License / Pixabay Content License / CC) or
  consented actor recording. License + source URL recorded below.
- Research datasets (TICaM/Drive&Act): eval-only, no redistribution, never
  shown publicly beyond their terms. SVIRO synthetic: demo-safe with
  attribution "SVIRO, DFKI".
- No clip is used for weight updates. Ever.
- Children in stock clips: platform model-released; prefer non-identifying
  framing and note it in the demo.

## Downloaded core set (2026-10-13, all Pexels License, landscape 1080p+)
| File | Scenario | Source page | Size | Notes |
|---|---|---|---|---|
| F1_family_ride.mp4 | Family ride, 2 adults front + baby rear, day | https://www.pexels.com/video/mom-and-dad-with-baby-inside-moving-car-6181413/ | 7.6 MB | golden-path candidate |
| C1_baby_carseat.mp4 | Dad strapping baby in child seat, belt fastening | https://www.pexels.com/video/man-strapping-his-baby-in-a-car-seat-6181408/ | 5.1 MB | child-seat + belt |
| C2_child_seat.mp4 | Child in car seat, day | https://www.pexels.com/video/a-child-smiling-at-the-camera-while-sitting-in-the-car-6300815/ | 4.9 MB | posture/face |
| E1_empty_cabin.mp4 | Empty parked BMW cabin pan, day | https://www.pexels.com/video/the-interior-of-a-parked-bmw-car-3757013/ | 4.0 MB | empty-cabin negative |
| N1_night_drive.mp4 | Driver POV night + streetlights, 40 s | https://www.pexels.com/video/man-driving-a-car-at-night-3554563/ | 27 MB | low-light test |
| S1_seatbelt_closeup.mp4 | Hands + belt close-up | https://www.pexels.com/video/close-up-footage-of-a-person-wearing-seatbelt-8155681/ | 1.3 MB | belt evidence |
| S2_belt_up.mp4 | Woman driver belts up, parked, 22 s | https://www.pexels.com/video/young-woman-putting-on-seat-belt-5391318/ | 11 MB | belt + driver |
| R1_rear_wide.mp4 | 4–5 rear passengers wide (taxi van) | https://www.pexels.com/video/group-journey-inside-a-busy-taxi-van-38684176/ | 5.6 MB | multi-occupant |

Refetched 2026-10-25 with `scripts/prepare_assets.py` (all eight clips verified
decodable). R1 resolved through the Pexels download endpoint at 3840×2160,
19.3 MB; S1 is 1280×720. Both remain within the license and resolution rules
above.

## Backup links (not yet downloaded)
| Scenario | Source page | Direct mp4 |
|---|---|---|
| Mom + baby rear interaction | https://www.pexels.com/video/mother-and-baby-playing-in-the-car-6181423/ | https://videos.pexels.com/video-files/6181423/6181423-hd_1920_1080_25fps.mp4 |
| Belt on kid seat close | https://www.pexels.com/video/dad-fastening-the-belt-on-his-kid-seat-6181415/ | https://videos.pexels.com/video-files/6181415/6181415-hd_1920_1080_25fps.mp4 |
| 2 kids belted rear | https://www.pexels.com/video/kids-wearing-their-seatbelts-4524519/ | https://videos.pexels.com/video-files/4524519/4524519-hd_1920_1080_25fps.mp4 |
| Empty luxury dash pan 45 s | https://www.pexels.com/video/interior-of-a-luxury-car-4994294/ | https://videos.pexels.com/video-files/4994294/4994294-hd_1920_1080_30fps.mp4 |
| Driver close-up day | https://www.pexels.com/video/video-of-a-man-inside-the-car-8470705/ | https://videos.pexels.com/video-files/8470705/8470705-hd_1920_1080_25fps.mp4 |
| Driver + passenger day | https://www.pexels.com/video/a-man-and-woman-in-a-car-4482238/ | https://videos.pexels.com/video-files/4482238/4482238-hd_1080_2048_25fps.mp4 |
| Night city drive POV 45 s | https://www.pexels.com/video/the-interior-of-a-car-at-night-with-lights-on-18616132/ | https://videos.pexels.com/video-files/18616132/18616132-hd_1920_1080_25fps.mp4 |
| Adult rear + laptop | https://www.pexels.com/video/a-businessman-closing-his-laptop-while-sitting-in-the-back-of-a-car-9520008/ | https://videos.pexels.com/video-files/9520008/9520008-hd_1366_720_25fps.mp4 |
| 2 adults rear bench | https://www.pexels.com/video/diverse-couple-inside-a-car-7154225/ | https://videos.pexels.com/video-files/7154225/7154225-hd_1920_1080_25fps.mp4 |

## Out-of-domain robustness clips (Mixkit License, landscape, verified by viewing)
| File | Source page | Direct mp4 | Content (verified, not assumed) |
|---|---|---|---|
| X1_street_crowd.mp4 | Busy street in the city | https://assets.mixkit.co/videos/4000/4000-720.mp4 | crowd ~45/frame; persons + cars/bikes; no seats claimed (uncalibrated) |
| X2_empty_office.mp4 | Open office space | https://assets.mixkit.co/videos/914/914-720.mp4 | BUSY office ~25 people (title misread — lesson: verify footage); stress test |
| X3_dog.mp4 | Dog being petted | https://assets.mixkit.co/videos/1478/1478-720.mp4 | dog correctly `dog`; partial person petting fires person (expected) |

## Datasets (eval-only)
| Dataset | Access | Status | Demo use |
|---|---|---|---|
| SVIRO synthetic rear-seat | https://sviro.kl.dfki.de/data/ (per-vehicle `?wpdmdl=` packages; bmw-i3 base = 2.6 GB) | DEFERRED 2026-10-14: no download budget; links + selection (bmw-i3 base) recorded; re-enactment list covers the gaps meanwhile | Safe with attribution |
| TICaM | https://vizta-tof.kl.dfki.de/cabin-dataset (request form) | Request to file; children are dolls | Synthetic subset preferred for public |
| Drive&Act | https://driveandact.com/ (direct, TB-scale) | Pull one NIR view on demand | No public raw clips without OK |
| DriveAHead | KIT email request | To file if head-pose needed | No public use |
| DMD / UTA-RLDD / YawDD / SFDDD | form / Drive / IEEE / Kaggle | DMS-lite eval candidates | Check per-clip publish flags |
