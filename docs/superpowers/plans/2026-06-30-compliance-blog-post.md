# Compliance Blog Post Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, Notion-styled standalone HTML blog post that tells the compliance-locomotion self-correction story, with reproducible figures, two comparison renders, and an Inter-versus-serif body font toggle for the user to choose.

**Architecture:** A single `blog/index.html` plus one hand-written `blog/style.css`, no framework and no build step. Figures are PNGs produced by committed scripts in `analysis/`, renders are GIFs produced by an `evaluation/` wrapper over the existing render tooling. A tiny font-toggle is the only JavaScript, present so the user can compare a serif body against an Inter body, and removable after they decide.

**Tech Stack:** HTML5, hand-written CSS, native `<details>`, matplotlib and pandas for figures, PIL plus stable-baselines3 for renders. Source Serif 4 body via Google Fonts, Geist sans self-hosted woff2, Inter via Google Fonts for the comparison toggle.

## Global Constraints

- All blog body prose uses only periods and commas. No em dashes, en dashes, semicolons, colons inside sentences, parentheses for asides, or exclamation points. Structural punctuation in tables, code, and URLs is exempt.
- Every quantitative claim traces to `results.md`, `FINDINGS.md`, or a committed CSV. No new experiments, no re-interpretation.
- Visual system follows the Notion blog spec, all-light, no dark hero band. Warm canvas `#f6f5f4`, white surfaces, hairline `#e6e6e6`, ink `#000`, single blue accent `#0075de` for links only.
- Policy labels in CSVs are `A_rigid`, `B_compliance`, `Bp_noh`, `A_notg`. Terrains are `T0_rigid, T1_firm, T2, T3, T4, T5, T6_train_edge, T7_extrap, T8_extrap, T9_extrap_soft`. Survival is `fall_rate < 0.5`.
- Env launch for any Python: `source .venv/bin/activate && PYTHONPATH="$PWD" python ...`.
- Work on branch `blog-post`. Commit after each task.

---

## File Structure

- `blog/index.html` — the post, one scrolling page, ten sections.
- `blog/style.css` — full Notion design system, serif body, sans headings, components.
- `blog/font-toggle.js` — minimal, toggles `body.font-inter` class, persists choice in localStorage. Temporary comparison aid.
- `blog/fonts/` — Geist woff2 files, self-hosted.
- `blog/figures/` — generated PNGs plus `avatar_placeholder.png`.
- `blog/renders/` — `A_t0_vs_t6.gif`, `Bprime_vs_B_t5.gif`.
- `analysis/plot_blog_figures.py` — Tier 1 data plots from the CSVs.
- `analysis/plot_schematics.py` — Tier 2 PMTG and stiffness-gauge schematics.
- `evaluation/render_blog.py` — Tier 3 wrapper producing the two comparison GIFs.

---

## Task 1: Scaffold blog directory, fonts, and CSS design system

**Files:**
- Create: `blog/style.css`, `blog/font-toggle.js`, `blog/fonts/` (Geist woff2), `blog/figures/avatar_placeholder.png`
- Create: `blog/index.html` (skeleton only, header plus empty section stubs)

**Interfaces:**
- Produces: CSS custom properties and component classes that Task 5 consumes. Class names, `.canvas`, `.reading-column`, `.post-header`, `.byline`, `.share-row`, `.share-btn`, `.figure`, `.figure figcaption`, `.claim-card`, `.claim-card .claim`, `.claim-card .test`, `.claim-card .broke`, `.scoreboard`, `.scoreboard.resolved`, `.pull-quote`, `details.setup-details`, and the body toggle `body.font-inter`.

- [ ] **Step 1: Acquire Geist woff2.** Download Geist sans (regular, medium, semibold) woff2 into `blog/fonts/`. If network is unavailable, fall back to a Google-Fonts sans that is not Inter, for example Space Grotesk via `<link>`, and note the substitution. Verify files exist and are non-zero.

- [ ] **Step 2: Write `blog/style.css`.** Define `:root` custom properties for the Notion palette, type scale, spacing 8px base, radii, and the layered micro-shadow. Set body to Source Serif 4 18 to 19px, line-height 1.65, ink color. Set headings, captions, byline, buttons to the Geist sans with negative tracking on large headings. Add `body.font-inter` override that swaps only the body reading font to Inter, headings stay sans. Implement component classes listed in Interfaces. Reading column max-width about 760px centered on the warm canvas, figures in white wells with 12px radius and ash captions, share buttons as pills, claim-card with three labeled rows and a restrained semantic red only on the `.broke` strike, scoreboard list with strike-on-`.resolved`, pull-quote large sans aside.

- [ ] **Step 3: Write `blog/font-toggle.js`.** About 10 lines. On click of a fixed-position toggle, add or remove `body.font-inter` and store the choice in localStorage, restoring it on load. Pure progressive enhancement, page is fully readable without it.

- [ ] **Step 4: Generate `avatar_placeholder.png`.** A simple monogram tile, initials DT, neutral background, via a 6-line matplotlib snippet or a small SVG rasterized. Verify the PNG exists and is square.

- [ ] **Step 5: Write `blog/index.html` skeleton.** Valid HTML5, links the stylesheet, the Google Fonts for Source Serif 4 and Inter, and `font-toggle.js` deferred. Include the post header with title, dek, byline row with avatar and placeholder affiliation and date, the share row with the four share-intent anchors against a `CANONICAL_URL` placeholder, the font-toggle control, and ten empty `<section>` stubs with ids matching the structure. Verify it opens in a browser and the design system is visibly applied.

- [ ] **Step 6: Commit.**

```bash
git add blog/ && git commit -m "feat(blog): scaffold Notion-styled shell, CSS system, font toggle"
```

---

## Task 2: Tier 1 data figures from the CSVs

**Files:**
- Create: `analysis/plot_blog_figures.py`
- Output: `blog/figures/hero_variance.png` (copied or re-rendered from existing `analysis/variance_plot.png`), `hero_variance_annotated.png`, `t0_performance.png`, `velocity_vs_robustness.png`, `contactvar_vs_robustness.png`, `speed_matched.png`, `convergence_orthogonality.png`

**Interfaces:**
- Consumes: `evaluation/ablation_results_multiseed.csv`, `evaluation/speed_matched_results.csv`, `evaluation/convergence_variance.csv`. Reuses the visual style of `analysis/plot_variance.py` for consistency.
- Produces: seven PNGs at the exact paths above, referenced by Task 5.

- [ ] **Step 1: Read `analysis/plot_variance.py`** to match colors, fonts, and figure sizing, so the new plots look like one family. Use the sticker-palette accent colors for the three policy lines, A_rigid, B_compliance, Bp_noh.

- [ ] **Step 2: hero_variance.** Reproduce the existing variance plot, fall_rate mean with 8-seed bands across T0 to T9 for the three 8-seed policies. If `analysis/variance_plot.png` already matches, copy it to `blog/figures/hero_variance.png`. Verify the survival counts implied at T5 and T6 reconcile with results.md, A 3/8 and 1/8, B 6/8 and 5/8, Bp_noh 8/8 and 7/8, by printing `(fall_rate<0.5).sum()` per policy per terrain.

- [ ] **Step 3: hero_variance_annotated.** Same plot with a shaded training-range span and an extrapolation-zone span past T6_train_edge, plus a vertical marker at the training edge. Verify the shading boundaries fall at the correct terrains.

- [ ] **Step 4: t0_performance.** Grouped bars at T0_rigid, forward_velocity_mean and cost_of_transport for A_rigid, B_compliance, Bp_noh. Verify A is fastest and lowest COT, matching the 0.21 m/s and COT 1.78 figures.

- [ ] **Step 5: velocity_vs_robustness.** Scatter, one point per seed, x is forward_velocity_mean averaged over transition terrains, y is number of terrains survived, colored by policy. Verify the A over B over Bp_noh speed ordering inverts against robustness.

- [ ] **Step 6: contactvar_vs_robustness.** Scatter, foot_contact_variance versus terrains survived per seed, colored by policy. Verify lower contact variance associates with more survival.

- [ ] **Step 7: speed_matched.** Grouped bars from `speed_matched_results.csv`, fall_rate at matched commanded velocity, brittle seeds versus robust seeds, showing the roughly 0.95 versus 0.00 gap on the transition terrain. Verify the gap is present.

- [ ] **Step 8: convergence_orthogonality.** Strip or dot plot, x is final `terrain_level_final2M` per seed for the B and Bp recipes from `convergence_variance.csv`, points colored by eval outcome robust or brittle from the multiseed CSV. Verify low-curriculum seeds appear among the robust, showing the decoupling, corr about +0.25.

- [ ] **Step 9: Run the script, confirm all seven PNGs exist and are non-empty.**

Run: `source .venv/bin/activate && PYTHONPATH="$PWD" python analysis/plot_blog_figures.py && ls -la blog/figures/*.png`
Expected: seven PNGs present, plus the printed reconciliation lines matching results.md.

- [ ] **Step 10: Commit.**

```bash
git add analysis/plot_blog_figures.py blog/figures/*.png && git commit -m "feat(blog): Tier 1 data figures, reconciled with results.md"
```

---

## Task 3: Tier 2 schematics

**Files:**
- Create: `analysis/plot_schematics.py`
- Output: `blog/figures/pmtg_trot_schematic.png`, `blog/figures/stiffness_gauge.png`

**Interfaces:**
- Produces: two PNGs referenced by Task 5. No data inputs, pure matplotlib drawing.

- [ ] **Step 1: pmtg_trot_schematic.** Top-down Go1 sketch, four legs, the diagonal pairs FR plus RL and FL plus RR marked as in-phase, the two pairs 180 degrees apart, and a small inset of the foot-path ellipse. Labels in the Geist-equivalent sans. Verify it reads as a trot diagram.

- [ ] **Step 2: stiffness_gauge.** A horizontal T0 to T9 scale, a spring or stiffness icon ramp from firm to soft, with the training edge marked at T6. A caption-ready note that the floors look identical and the physics parameter is what changes. Verify the ramp direction matches firmer at T0.

- [ ] **Step 3: Run, confirm both PNGs exist.**

Run: `source .venv/bin/activate && PYTHONPATH="$PWD" python analysis/plot_schematics.py && ls -la blog/figures/pmtg_trot_schematic.png blog/figures/stiffness_gauge.png`
Expected: both present and non-empty.

- [ ] **Step 4: Commit.**

```bash
git add analysis/plot_schematics.py blog/figures/pmtg_trot_schematic.png blog/figures/stiffness_gauge.png && git commit -m "feat(blog): Tier 2 PMTG and stiffness-gauge schematics"
```

---

## Task 4: Tier 2 comparison renders

**Files:**
- Create: `evaluation/render_blog.py` (thin wrapper over `evaluation/render_comparison.py` internals)
- Output: `blog/renders/A_t0_vs_t6.gif`, `blog/renders/Bprime_vs_B_t5.gif`

**Interfaces:**
- Consumes: checkpoints `checkpoints/policy_a_v24/policy_v24_final.zip` and its seed variants, `checkpoints/policy_b/policy_b_final.zip` and variants, `checkpoints/policy_b_noh/policy_b_noh_final.zip` and variants, with matching `vecnorm_final.pkl`. The terrain suite `evaluation/terrain_suite.py` TERRAINS and `EvalCompliantEnv`. The rollout and panel-compositing helpers in `render_comparison.py`.
- Produces: two downscaled GIFs at the paths above.

- [ ] **Step 1: Pick seeds from the eval.** From the multiseed CSV, select a robust Bp_noh or B_compliance seed, lowest transition fall_rate, and a brittle B_compliance seed, highest transition fall_rate at T5, for the `Bprime_vs_B_t5` comparison. Print the chosen seeds.

- [ ] **Step 2: A_t0_vs_t6.** Roll Policy A_rigid deterministically on T0_rigid, left panel, and on T6_train_edge, right panel, same seed and physics, label each with policy, terrain, and FELL@step. Compose side-by-side, downscale, write `blog/renders/A_t0_vs_t6.gif`.

- [ ] **Step 3: Bprime_vs_B_t5.** Roll the robust B-prime seed, left, and the brittle B seed, right, both on T5, same physics seed, label and compose, write `blog/renders/Bprime_vs_B_t5.gif`. Use the correct obs_mode per policy, 49D for Bp_noh and A, 89D for B.

- [ ] **Step 4: Run and confirm both GIFs exist, are non-empty, and are web-sized, under a few MB each.**

Run: `source .venv/bin/activate && PYTHONPATH="$PWD" python evaluation/render_blog.py && ls -la blog/renders/*.gif`
Expected: two GIFs present, the A clip visibly shows confident walk then a fall, the B-prime clip shows the steadier gait surviving while B struggles.

- [ ] **Step 5: Commit.**

```bash
git add evaluation/render_blog.py blog/renders/*.gif && git commit -m "feat(blog): A faceplant and B-prime versus B comparison renders"
```

---

## Task 5: Write the post content and wire all assets

**Files:**
- Modify: `blog/index.html` (fill the ten sections, embed figures and renders, instantiate the components)

**Interfaces:**
- Consumes: every class from Task 1, every PNG from Tasks 2 and 3, both GIFs from Task 4.
- Produces: the finished post.

- [ ] **Step 1: Scoreboard teaser, section 2.** Instantiate the `.scoreboard` component, five tempting claims listed, history is required, the clean efficiency win, A is 1/8 reliable, convergence predicts robustness, slower is more robust, with the survivor hidden or muted. Write the intro lines in the constrained prose.

- [ ] **Step 2: Cold open, section 3.** Write the first clean-but-wrong result and the thesis, about 350 to 450 words.

- [ ] **Step 3: Setup, section 4.** Trimmed prose plus the two schematics, with deeper methodology inside `details.setup-details`. Honest note about identical-looking floors. About 400 words visible.

- [ ] **Step 4: Headline tradeoff, section 5.** Prose plus `hero_variance`, `t0_performance`, and the `A_t0_vs_t6` render. State the monotonic ordering and the at-home win for A. About 500 words.

- [ ] **Step 5: The arc, section 6.** Three claim-test-broke cards, provenance bug, convergence, speed, each with its figure, `convergence_orthogonality`, `velocity_vs_robustness`, `speed_matched`. Connective prose between cards. About 700 words.

- [ ] **Step 6: The reversal, section 7.** Short standalone section, foot history did not help, with the `Bprime_vs_B_t5` render as proof. About 300 words.

- [ ] **Step 7: Mechanism, section 8.** Gait quality, `contactvar_vs_robustness`, measurable not yet inducible. About 400 words.

- [ ] **Step 8: Boundary, section 9.** No extrapolation, `hero_variance_annotated`. About 300 words.

- [ ] **Step 9: What it means, section 10.** Methodology lessons as pull-quotes, brief future work, the resolved scoreboard with strikes applied and survivor highlighted, one-line close. About 500 words.

- [ ] **Step 10: Verify assets and reconciliation.** Open the page, confirm every `<img>` and figure resolves, both GIFs play, the toggle flips body font between serif and Inter, and spot-check three numbers against results.md.

- [ ] **Step 11: Commit.**

```bash
git add blog/index.html && git commit -m "feat(blog): full post content, components, and wired figures"
```

---

## Task 6: Final integration, lint, and review

**Files:**
- Modify: any file needing fixes found during review.
- Create: `blog/README.md` (one short paragraph, how to rebuild figures and renders, and the publish placeholders).

- [ ] **Step 1: Punctuation lint.** Grep the visible prose of `index.html` for disallowed marks. Confirm no em dashes, en dashes, semicolons, parenthetical parens, or stray colons inside sentences appear in body copy. Fix any.

Run: `grep -nE '—|–|;|\([a-z]' blog/index.html` then manually clear true prose hits.

- [ ] **Step 2: Word count.** Confirm the body prose lands between 3000 and 4500 words. Adjust if short or long.

- [ ] **Step 3: Placeholders marked.** Confirm avatar, affiliation, and `CANONICAL_URL` are present and clearly marked for the user.

- [ ] **Step 4: Browser check.** Open `blog/index.html`, scroll the full post, confirm the Notion all-light look, the serif-body sans-heading pairing, the claim-card rhythm, and the scoreboard bookends. Capture a screenshot for the user.

- [ ] **Step 5: Write `blog/README.md`** with the rebuild commands and the publish-time placeholders.

- [ ] **Step 6: Commit and report.**

```bash
git add blog/ && git commit -m "chore(blog): lint, README, final integration"
```

---

## Self-Review notes

- Spec coverage. Sections 1 through 11 of the spec map to Tasks 1 through 6, structure to Task 5, figures to Tasks 2 and 3, renders to Task 4, components to Tasks 1 and 5, fonts and the Inter comparison to Task 1, punctuation and word-count gates to Task 6.
- The Inter-versus-serif comparison requested by the user is implemented as the Task 1 font toggle, default serif, so the user can decide by clicking rather than diffing two files.
- A_notg has only 2 seeds and is not a figure, it is a prose mention in the setup, consistent with the spec caveat.
