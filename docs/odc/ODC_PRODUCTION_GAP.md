# ODC production-gap plan (v1, locked 2026-10-13)

Rule: every row names an owner and an evidence plan. Nothing here is claimed.

| Area | Today | Gap to production | Owner | Evidence plan |
|---|---|---|---|---|
| Functional safety (ISO 26262) | QM demonstrator, advisory outputs only | HARA, ASIL capability (target ASIL A/B for linked functions), SEooC safety case, FMEDA, MISRA, traceability | OEM + ODC | safety plan + work products per phase gate |
| Process (ASPICE) | phase-gated, tested, documented | CL2 managed (SYS/SWE/VAL/SUP/MAN), then CL3; ACQ.4 supplier monitoring | ODC | process gap assessment pre-pilot |
| Cybersecurity (ISO/SAE 21434) | loopback, no auth, no TLS (engine) | TARA, secure boot/OTA, CSMS/SOMS evidence, vuln monitoring | ODC + PLOMID | TARA workshop; transport decision |
| Privacy/legal (GDPR/PIPL) | metadata-only, no identity, closed-loop | DPIA, retention schedules, consent/HMI notice, cross-border plan | Owner legal | DPIA template + review |
| Software lifecycle | locked plan, versioned config/models, regression suite | long-term maintenance, EOL, defect SLA, OTA policy | ODC | lifecycle doc pre-pilot |
| Hardware qualification | laptop reference only | AEC-Q target grade, temp/vibration profiles, qualified SoC list | OEM | qual plan with vehicle program |
| Model maintenance | pinned checkpoints, parity gates | drift monitoring, re-validation cadence, pack release process | ODC | MLOps-lite runbook |
| Supplier rights | Apache/MIT/BSD deps scanned; ckpt BOM per artifact | full SBOM + redistribution rights per checkpoint | Owner legal | BOM sign-off per release |
| Validation coverage | 22 hand frames + 8 clips | SVIRO/TICaM scale-up, NIR/fisheye buckets, misuse renders | ODC | validation plan phases |

Entry criterion for any production claim: this table has no GAP rows
without an accepted, dated mitigation. Until then the product label stays
"engineering demonstrator and pilot platform".
