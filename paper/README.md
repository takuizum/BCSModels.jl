# Paper draft (JOSS submission)

This directory holds the draft submission to the
[Journal of Open Source Software (JOSS)](https://joss.theoj.org).

Files:

- `paper.md` — the manuscript itself (Markdown with a YAML front matter
  that JOSS's editorial bot understands).
- `paper.bib` — BibTeX bibliography.

## How JOSS expects the paper

JOSS papers are short (about 250–1000 words) and focused on **what the
software does, why it is needed, and the academic/engineering context**.
The methodology *itself* lives elsewhere (in our case, the arXiv preprint
and the `docs/theory.md` implementation reference); the JOSS paper points
to it.

## Workflow

1. **First:** put the methodology preprint on arXiv. Replace the bracketed
   ORCID and affiliation placeholders in `paper.md` once you have them.
   Add the arXiv reference to `paper.bib` (key: `izumi_bcsm_arxiv`) and
   cite it from the Summary section.
2. **Then:** publish a tagged release (e.g. `v0.1.0`) on GitHub and
   archive it on [Zenodo](https://zenodo.org). Zenodo will mint a DOI;
   record it in the JOSS submission form.
3. **Then:** open an issue on
   [`openjournals/joss-reviews`](https://github.com/openjournals/joss-reviews)
   via the JOSS submission form. The form asks for the repo URL, the
   Zenodo DOI, and the path to `paper.md` inside the repo (`paper/paper.md`).
4. **Review:** JOSS assigns an editor and two reviewers. Reviewers test
   the install, run the examples, and check that the paper's claims match
   the code. Average turnaround is 4–8 weeks.
5. **Publication:** JOSS mints a DOI for the paper itself
   (`10.21105/joss.XXXXX`). Update `CITATION.cff` to promote that DOI to
   `preferred-citation`.

## Local preview

You can preview the rendered PDF using the official JOSS docker image:

```bash
docker run --rm \
    --volume $PWD/paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
```

The rendered PDF appears at `paper/paper.pdf`.
