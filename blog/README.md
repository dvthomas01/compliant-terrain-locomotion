# Blog post, Five clean results, five times wrong

A self contained, static blog post. Open `index.html` in any browser, no build step.
Styled after the Notion blog, all light, with a serif reading body and a sans heading
face. The body font toggle in the corner flips between Source Serif 4 and Inter so the
author can pick a final body font. It is a temporary aid and can be removed once chosen.

## Files

- `index.html`, the post.
- `style.css`, the Notion style design system and components.
- `font-toggle.js`, the serif versus Inter body toggle, progressive enhancement only.
- `figures/`, generated data plots and schematics, plus the avatar placeholder.
- `renders/`, the two comparison GIFs.

## Rebuilding the assets

From the repo root, with the project venv active.

    source .venv/bin/activate

    # Tier 1 data figures, from the committed eval CSVs
    PYTHONPATH="$PWD" python analysis/plot_blog_figures.py

    # Tier 2 schematics
    PYTHONPATH="$PWD" python analysis/plot_schematics.py

    # Trot diagram, real Go1 top and side renders plus the phase clock
    PYTHONPATH="$PWD" python analysis/render_trot_diagram.py

    # Comparison renders, A faceplant and B prime versus B
    PYTHONPATH="$PWD" python evaluation/render_blog.py

Every number in the post is read from `evaluation/ablation_results_multiseed.csv`,
`evaluation/speed_matched_results.csv`, and `evaluation/convergence_variance.csv`, so
the figures reconcile with `results.md` by construction. The figure script prints the
survival reconciliation when it runs.

## Before publishing

Three placeholders are marked in `index.html`.

- The avatar. The byline points at `figures/avatar.jpg`, which currently holds a
  monogram fallback. Overwrite it with the author photo, square crops best.
- The affiliation line in the byline, marked with the `placeholder` class.
- `CANONICAL_URL` in the four share links, set it to the published post URL.

## Local preview

    cd blog
    python3 -m http.server 8731

Then open `http://localhost:8731/index.html`. Serving over http rather than the file
protocol lets the Google Fonts load.
