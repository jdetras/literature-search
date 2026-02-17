# Open Access Literature Search & Review

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/jdetras/literature-search)
[![Last Commit](https://img.shields.io/github/last-commit/jdetras/literature-search)](https://github.com/jdetras/literature-search/commits)
[![Stars](https://img.shields.io/github/stars/jdetras/literature-search?style=social)](https://github.com/jdetras/literature-search/stargazers)

A Streamlit app for searching open‑access literature across multiple sources, scoring relevance, and exporting results in Markdown and JSON with APA citations.

**Demo:** [Literature Search Streamlit webappL]https://literature-search.streamlit.app/)

## Features
- Query by topic and organism(s)
- Year range filter
- Relevance threshold (0–1)
- Optional journal/venue filter (e.g., `Nature, Science`)
- Sources: Europe PMC, DOAJ, Semantic Scholar, Crossref, OpenAlex (OA), arXiv, bioRxiv
- Outputs: on‑page results + downloadable Markdown and JSON

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and click **New app**.
3. Select your repo and set:
   - **Main file path**: `app.py`
   - **Python version**: default is fine
4. Deploy.

## Usage Tips
- If you get very few results, try:
  - clearing the journal filter
  - lowering the relevance threshold (e.g., `0.05`)
  - increasing max results
  - widening the year range

## Notes
- bioRxiv API does not support keyword search; results are pulled by date range and then filtered by relevance.
- OpenAlex results are restricted to open access (`open_access.is_oa:true`), which includes hybrid OA.

## Configuration
- Streamlit config is in `.streamlit/config.toml`.

## Data Sources
- Europe PMC
- DOAJ
- Semantic Scholar
- Crossref
- OpenAlex (OA)
- arXiv
- bioRxiv

## Limitations
- Relevance scoring is TF‑IDF on title + abstract; it is not a semantic embedding model.
- Some records may not include abstracts or full‑text links, depending on source metadata.
