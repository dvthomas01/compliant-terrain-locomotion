# Design — Compliance-Locomotion Blog Post

> Created 2026-06-30. Status: design (pre-implementation).
> Source content: `results.md` (authoritative numbers), `retrospective.md` (narrative),
> `FINDINGS.md` (one-screen summary). Visual system: the Notion blog design spec
> (warm-paper canvas, near-black Inter, single blue accent).

## 1. Goal

Produce a polished, self-contained **standalone HTML blog post** that tells the
compliance-locomotion ablation story to a technically literate audience, structured
around the project's **self-correction arc** ("clean results that turned out wrong"),
and styled after the Notion blog (with the TRI all-light layout preference).

This is a writing and frontend-presentation task. The science is settled and
documented. We are not changing any result. Every number in the post must trace to
`results.md`, `FINDINGS.md`, or the committed CSVs.

## 1a. Writing constraint (hard rule)

All prose in the blog body uses **only periods and commas** as punctuation. No em
dashes, no en dashes, no semicolons, no colons inside sentences, no parentheses for
asides, no exclamation points. Lists are introduced with a period and broken into
short sentences instead. Structural punctuation outside prose, such as table pipes,
code, and URLs, is exempt. This rule also governs assistant responses for this work.

## 2. Confirmed decisions

| Decision | Choice |
|---|---|
| Audience | Technical peers and researchers, RL and robotics literate |
| Spine | The self-correction arc, "five clean results, five times wrong" |
| Length | Long-form, about 3000 to 4500 words |
| Visual system | Notion blog spec, all-light, no dark-indigo hero band |
| Typography | Serif body and sans headings, NOT all-Inter, see section 4 |
| Byline | "Dami Thomas" plus placeholder affiliation line and placeholder avatar |
| Renders | Generate now via `render_comparison.py`, extended |
| Home | New in-repo `blog/` directory, no build step |

## 3. Deliverable & file layout

```
blog/
├── index.html          # the post (single page, semantic HTML)
├── style.css           # Notion design system, hand-written, no framework
├── figures/            # generated data plots + schematics (PNG) + avatar placeholder
│   ├── hero_variance.png
│   ├── hero_variance_annotated.png
│   ├── t0_performance.png
│   ├── velocity_vs_robustness.png
│   ├── contactvar_vs_robustness.png
│   ├── speed_matched.png
│   ├── convergence_orthogonality.png
│   ├── pmtg_trot_schematic.png
│   ├── stiffness_gauge.png
│   └── avatar_placeholder.png   # monogram tile; user swaps in a photo
└── renders/            # comparison GIFs (downscaled, web-sized)
    ├── A_t0_vs_t6.gif       # A confident (T0) vs A faceplant (T6)
    └── Bprime_vs_B_t5.gif   # B′ (robust) vs B (brittle) on soft terrain
```

Figure-generation scripts live in `analysis/` (reproducible, committed) — NOT inline
in HTML. The post references the PNG/GIF outputs.

- New plot scripts: `analysis/plot_blog_figures.py` (Tier 1 data plots, reads the
  committed CSVs) and `analysis/plot_schematics.py` (Tier 2 PMTG + stiffness-gauge
  schematics, pure matplotlib, no data).
- Renders: extend/parametrize `evaluation/render_comparison.py` (or a thin
  `evaluation/render_blog.py` wrapper) to emit the two blog GIFs, then downscale into
  `blog/renders/`.

## 4. Visual system (Notion spec → CSS variables)

Faithful subset of the pasted Notion spec. All-light (the one documented exception,
the dark hero band, is **omitted** per user choice).

- **Canvas** `--canvas-soft: #f6f5f4` (page background). **Surface** `#ffffff` for
  figure wells / cards. **Hairline** `#e6e6e6` 1px borders.
- **Ink** `#000` (~95% alpha) headings/body; `--ink-secondary #31302e`;
  `--ink-muted #615d59`; `--ink-faint #a39e98` for captions/metadata.
- **Accent** `--primary #0075de` — **links only**. No other structural color.
- **Type, deliberate serif-plus-sans pairing.** Inter is not used for body. Inter is
  a UI face, it tires at length, and it is the most overused font on technical blogs.
  Body is set in a reading serif, **Source Serif 4** (Google Fonts, with a
  `Newsreader` alternative noted for review), at 18 to 19px, 400 weight, 1.6 to 1.7
  line height for comfortable long-form reading. Headings, figures, captions, and all
  UI chrome use a clean distinctive sans, **Geist** (self-hosted woff2 in `blog/fonts/`,
  OSS, clearly not Inter). This serif-body, sans-heading split is the classic
  research-reading pairing and keeps the post feeling considered rather than defaulted.
  Headings stay 600 to 700 with tight negative tracking, H1 about 40px, H2 about 26px.
  Weight and serif-versus-sans contrast carry the hierarchy. Final font choice is open
  for the user to confirm at spec review.
- **Shape**: figure wells `border-radius: 12px`; share buttons pill `9999px`.
- **Elevation**: hairline + barely-there layered micro-shadow only; no heavy drops.
- **Reading column**: centered, ~720–820px max-width, generous outer gutters and
  large vertical section gaps (whitespace as the grouping device).
- **Sticker palette**: appears **only inside the data figures** (matplotlib line/marker
  colors), never as page chrome.

**Header and byline block.** TRI and Notion style, on the light canvas. Post title in
the large heading face with tight tracking, a one-line dek beneath it, then a row with
the avatar, "Dami Thomas", a placeholder affiliation line, and the date. A set of pill
share buttons follows for X, LinkedIn, Facebook, and Email. Share links use standard
share-intent URLs against a `CANONICAL_URL` placeholder constant that the user sets on
publish.

## 5. Page structure (information architecture)

Single scrolling page. The arc is restructured per review notes to keep momentum,
hook then tradeoff then arc, with the heavy setup trimmed and the emotional peak
isolated. The signature visual rhythm is a repeating claim-test-broke card, defined in
section 6a.

1. **Header.** Title, dek, byline and share row. Light, no dark band.
2. **Scoreboard teaser.** The opening tally graphic. Five tempting claims listed,
   suspense held. Pays off at the close. See section 6a.
3. **Cold open.** The first clean-but-wrong result, the single-seed story that
   history helps with an apparent efficiency win, undone by seed variance and the
   frozen-robot-counts-as-survivor metric bug. States the thesis, a story about
   catching clean stories before they become claims.
4. **The setup, trimmed.** Only the minimum needed for section 5. Go1, MuJoCo,
   solver-level Level-1 compliance, the A, B, B-prime ablation with identical recipe
   and only terrain and observation differing, and the T0 to T9 softness sweep. Deeper
   methodology lives in an inline collapsible `<details>` block, "Setup details", so it
   does not stall the read. Figures, `pmtg_trot_schematic` and `stiffness_gauge`. Honest
   note, the floors look identical, the physics parameter is what changes.
5. **The headline tradeoff.** Monotonic B-prime over B over A. A is fastest and most
   efficient at home and fails categorically past the training edge. Figures,
   `hero_variance` and `t0_performance`, plus render `A_t0_vs_t6`.
6. **The arc of being wrong, the engine.** Three supporting refutations, each rendered
   as a claim-test-broke card. The `train_policy_a.py` provenance bug, the
   "A is 1/8 reliable" claim that was retracted. Convergence predicts robustness,
   refuted. Slower-is-more-robust, correlational then refuted by the speed-matched
   intervention. Figures, `convergence_orthogonality`, `velocity_vs_robustness`,
   `speed_matched`.
7. **The reversal.** A short standalone section for the emotional peak, promoted out of
   the list. Foot history did not help, it slightly hurt, reversing the central
   hypothesis. Render `Bprime_vs_B_t5` lands here as the visual proof of the gait
   difference.
8. **The mechanism.** Robustness is a soft-contact-stable gait with lower foot-contact
   variance. Measurable, not yet inducible by any single knob. Figure,
   `contactvar_vs_robustness`.
9. **The boundary.** No free extrapolation, a narrow T7 margin then collapse by T8 to
   T9. Figure, `hero_variance_annotated` with training-range versus extrapolation-zone
   shading.
10. **What it means.** Methodology lessons surfaced as pull-quotes, single-seed is not
    evidence, provenance discipline matters, test the tempting story, metrics encode
    assumptions, negative results are the product. Brief future work. The closing
    scoreboard, five claims struck through and the one survivor standing, the
    gait-quality mechanism. One-line close.

## 6. Figure manifest

### Tier 1 — data plots (from committed CSVs)

Data sources: `evaluation/ablation_results_multiseed.csv`
(cols: policy, seed, terrain, forward_velocity_mean, fall_rate, success_rate,
stall_rate, cost_of_transport, foot_contact_variance, base_height_variance),
`evaluation/speed_matched_results.csv` (seed, terrain, command, fall_rate, velocity),
`evaluation/convergence_variance.csv` (recipe, seed, terrain_level_final2M, ep_len_final2M).

| Figure | Content | Source |
|---|---|---|
| `hero_variance` | Reuse existing `analysis/variance_plot.png` (fall-rate vs T0–T9, 3 policies, 8-seed bands) | existing `plot_variance.py` |
| `hero_variance_annotated` | Same plot + shaded training-range vs extrapolation-zone | multiseed CSV |
| `t0_performance` | Grouped bars: velocity + COT for A/B/B′ at T0 | multiseed CSV |
| `velocity_vs_robustness` | Scatter: per-seed velocity (x) vs #terrains-survived / transition fall-rate (y), colored by policy | multiseed CSV |
| `contactvar_vs_robustness` | Scatter: foot-contact-variance (x) vs robustness (y), per seed | multiseed CSV |
| `speed_matched` | Grouped bars: fall rate at matched velocity, brittle vs robust seeds (~0.95 vs ~0.00) | speed_matched CSV |
| `convergence_orthogonality` | Strip/dot plot: final terrain_level per seed (B, B′), colored by eval outcome (robust/brittle) | convergence + multiseed CSVs |

All numbers must reconcile with `results.md`. If a derived quantity (e.g.
"#terrains survived") isn't directly in a CSV, compute it from `fall_rate < 0.5`
exactly as `results.md` defines survival.

### Tier 2 — schematics (no data)

| Figure | Content |
|---|---|
| `pmtg_trot_schematic` | Go1 top-down: diagonal leg pairs (FR+RL, FL+RR) 180° out of phase + foot-path ellipse; conveys "propulsive by construction" |
| `stiffness_gauge` | T0→T9 horizontal scale with a stiffness/spring icon ramp; "floors look identical, physics differs" |

### Tier 3 — renders (generate now)

Via extended `render_comparison.py`. Deterministic single-episode rollouts, same
seed/physics, downscaled GIFs.

| Render | Left | Right | Terrain |
|---|---|---|---|
| `A_t0_vs_t6` | Policy A on T0 (confident walk) | Policy A on T6 (faceplant) | T0 vs T6 |
| `Bprime_vs_B_t5` | B′ (robust seed) | B (brittle seed) | T5 (transition) |

Checkpoints (verified present): `checkpoints/policy_a_v24/policy_v24_final.zip`,
`checkpoints/policy_b/policy_b_final.zip`,
`checkpoints/policy_b_noh/policy_b_noh_final.zip` (+ seed variants s1–s7). Choose a
representative robust B′ seed and brittle B seed from the multiseed eval for the
side-by-side. GIFs downscaled/subsampled (existing script already does PANEL_W=320,
SUBSAMPLE=2) and copied into `blog/renders/`.

## 6a. Signature components

Three reusable HTML and CSS components carry the post's identity. All three are
hand-built markup, not generated images, so they stay crisp and themeable.

- **Claim-test-broke card.** The post's signature, repeated across sections 3, 6, and
  7. A card with three labeled rows. The claim, stated as the tempting clean story. The
  test, the thing we ran to check it. What broke, the outcome, marked with a small
  muted strike treatment. This is the one place a restrained semantic red appears, on
  the strike or the cross glyph only, used sparingly as content and never as page
  chrome, so the blue-only accent discipline still holds for UI. Consistent layout
  across every instance is the whole point, it becomes the visual rhythm of the read.
- **Scoreboard.** A tally rendered in HTML, five tempting claims struck through and the
  single survivor standing, the soft-contact-stable-gait mechanism. Appears twice, a
  teaser near the top with claims unstruck and held in suspense, and a resolution at
  the close with the strikes applied and the survivor highlighted. The two states share
  one component, the close just toggles a class.
- **Pull-quote.** A large-type aside pulled from the methodology lessons in section 10,
  set in the sans face, for skimmable thesis lines such as "Single-seed is not
  evidence" and "Negative results are the product". Gives the long read texture and a
  scan path.

## 7. Tech approach

- Plain semantic HTML5 and one hand-written `style.css`. No framework and no build
  step. Opens via `file://` or any static host. JavaScript is optional progressive
  enhancement only, see section 10.
- Fonts. Source Serif 4 body via Google Fonts `<link>`, Geist sans self-hosted as
  woff2 in `blog/fonts/`, both with a system-stack fallback so the page still reads
  offline.
- Share buttons. Anchor tags with standard share-intent URLs. `CANONICAL_URL` is a
  clearly-marked placeholder the user sets at publish time.
- Images. Local relative paths into `figures/` and `renders/`, inside rounded white
  wells with ash-grey `<figcaption>` captions.
- Native disclosure. The "Setup details" block uses the native HTML `<details>`
  element, so progressive disclosure works with zero JavaScript.
- Accessibility basics. Alt text on every figure, sufficient contrast, semantic
  headings.
- Responsive. Single-column reading layout, figures scale to container, comfortable on
  mobile. The Notion breakpoints simplify to "reading column shrinks, figures go
  full-width".

## 8. Content sourcing & accuracy constraint

Prose is newly written for the blog voice, but **all quantitative claims are lifted
from `results.md` / `FINDINGS.md`** (8-seed survival counts, velocities, COT, the
speed-matched 0.95-vs-0.00, correlations +0.25 / −0.40). The retracted/negative
results are stated as part of the arc, matching the transparency already in those
docs. No new experiments; no re-interpretation.

## 9. Placeholders the user will fill

- **Avatar**: `figures/avatar_placeholder.png` (monogram tile) — swap for a photo.
- **Affiliation line**: placeholder text in the byline.
- **`CANONICAL_URL`**: post URL for share links, set at publish.

## 10. Out of scope (v1) and optional stretch

Out of scope for v1.

- No static-site generator, blog framework, or RSS.
- No new RL experiments or metric changes.
- No dark mode or dark hero band.
- No video or mp4 embedding. GIFs only, to stay markdown and static-host friendly.
- Level-2 geometric compliance, sim-to-real, and other future-work items remain prose
  mentions only.

Optional stretch, deferred, only if we relax the no-JavaScript default.

- A T0 to T9 softness slider tied to the variance plot, as pure progressive
  enhancement. Static image by default, interactive if JavaScript runs. This would need
  per-step plot states, so it is explicitly a post-v1 nice-to-have and does not block
  the core build.

## 11. Success criteria

1. `blog/index.html` opens standalone and renders the full post with all figures and
   both GIFs, visibly in the Notion all-light style with the serif-body sans-heading
   pairing.
2. Every number in the post reconciles with `results.md` and the CSVs.
3. The ten-section arc reads as a coherent self-correction narrative, about 3000 to
   4500 words, in the technical-peer voice, with the repeating claim-test-broke card as
   its visual rhythm and the scoreboard bookending it.
4. All figures and renders are reproducible from committed scripts, checkpoints, and
   CSVs.
5. The byline and share row are present, with the three placeholders clearly marked.
6. All blog prose uses only periods and commas, per section 1a.
