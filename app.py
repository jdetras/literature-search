import concurrent.futures
import datetime as dt
import json
import math
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


USER_AGENT = "oa-literature-agent/1.0 (+https://example.org)"
TIMEOUT = 20


class Paper:
    def __init__(
        self,
        title: str,
        abstract: str,
        year: Optional[int],
        authors: List[str],
        journal: Optional[str],
        doi: Optional[str],
        url: Optional[str],
        full_text_url: Optional[str],
        source: str,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
    ):
        self.title = title
        self.abstract = abstract
        self.year = year
        self.authors = authors
        self.journal = journal
        self.doi = doi
        self.url = url
        self.full_text_url = full_text_url
        self.source = source
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.relevance = 0.0

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": self.authors,
            "journal": self.journal,
            "doi": self.doi,
            "url": self.url,
            "full_text_url": self.full_text_url,
            "source": self.source,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "relevance": round(self.relevance, 4),
            "apa_citation": apa_citation(self),
            "summary": summarize_abstract(self.abstract),
        }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def summarize_abstract(abstract: str) -> str:
    abstract = normalize_text(abstract)
    if not abstract:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    return " ".join(sentences[:3]).strip()


def apa_author(author: str) -> str:
    parts = author.split()
    if not parts:
        return author
    last = parts[-1]
    initials = " ".join([p[0].upper() + "." for p in parts[:-1] if p])
    if initials:
        return f"{last}, {initials}"
    return last


def apa_citation(paper: Paper) -> str:
    authors = paper.authors[:6]
    author_str = ", ".join(apa_author(a) for a in authors)
    if len(paper.authors) > 6:
        author_str += ", et al."
    year = paper.year or "n.d."
    title = paper.title.rstrip(".") + "."
    journal = paper.journal or ""
    vol = paper.volume or ""
    issue = f"({paper.issue})" if paper.issue else ""
    pages = paper.pages or ""
    parts = [author_str, f"({year}).", title]
    if journal:
        parts.append(journal)
    if vol or issue:
        parts.append(f"{vol}{issue}")
    if pages:
        parts.append(pages)
    if paper.doi:
        parts.append(f"https://doi.org/{paper.doi}")
    return " ".join([p for p in parts if p]).strip()


def tfidf_relevance(query: str, papers: List[Paper]) -> None:
    texts = [query] + [normalize_text(p.title + " " + (p.abstract or "")) for p in papers]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    query_vec = tfidf[0:1]
    doc_vecs = tfidf[1:]
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    for p, s in zip(papers, sims):
        p.relevance = float(s)


def filter_by_year(papers: List[Paper], start_year: int, end_year: int) -> List[Paper]:
    out = []
    for p in papers:
        if p.year is None:
            continue
        if start_year <= p.year <= end_year:
            out.append(p)
    return out


def arxiv_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = int(published[:4]) if published[:4].isdigit() else None
        if year and not (start_year <= year <= end_year):
            continue
        authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
        authors = [a for a in authors if a]
        landing = None
        pdf_url = None
        for link in entry.findall("atom:link", ns):
            rel = link.attrib.get("rel")
            ltype = link.attrib.get("type")
            href = link.attrib.get("href")
            if rel == "alternate" and href:
                landing = href
            if ltype == "application/pdf" and href:
                pdf_url = href
        results.append(
            Paper(
                title=title,
                abstract=abstract,
                year=year,
                authors=authors,
                journal="arXiv",
                doi=None,
                url=landing,
                full_text_url=pdf_url,
                source="arXiv",
            )
        )
    return results


def biorxiv_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    cursor = 0
    results: List[Paper] = []
    while len(results) < limit:
        url = f"https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        collection = data.get("collection", [])
        if not collection:
            break
        for item in collection:
            title = item.get("title") or ""
            abstract = item.get("abstract") or ""
            date_str = item.get("date") or ""
            year = int(date_str[:4]) if date_str[:4].isdigit() else None
            if year and not (start_year <= year <= end_year):
                continue
            authors = [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()]
            doi = item.get("doi")
            landing = f"https://doi.org/{doi}" if doi else None
            results.append(
                Paper(
                    title=title,
                    abstract=abstract,
                    year=year,
                    authors=authors,
                    journal="bioRxiv",
                    doi=doi,
                    url=landing,
                    full_text_url=item.get("link"),
                    source="bioRxiv",
                )
            )
            if len(results) >= limit:
                break
        cursor += len(collection)
    return results


def europe_pmc_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    qp = f"({query}) AND OPEN_ACCESS:Y AND PUB_YEAR:[{start_year} TO {end_year}]"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": qp, "format": "json", "pageSize": limit, "sort": "CITED"}
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("resultList", {}).get("result", []):
        authors = [a.strip() for a in (item.get("authorString") or "").split(",") if a.strip()]
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        landing = item.get("doiUrl")
        if not landing and pmid:
            landing = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if not landing and pmcid:
            landing = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        results.append(
            Paper(
                title=item.get("title") or "",
                abstract=item.get("abstractText") or "",
                year=int(item.get("pubYear")) if item.get("pubYear") else None,
                authors=authors,
                journal=item.get("journalTitle"),
                doi=item.get("doi"),
                url=landing,
                full_text_url=item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url"),
                source="Europe PMC",
                volume=item.get("journalVolume"),
                issue=item.get("issue"),
                pages=item.get("pageInfo"),
            )
        )
    return results


def openalex_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    url = "https://api.openalex.org/works"
    filters = [
        f"from_publication_date:{start_year}-01-01",
        f"to_publication_date:{end_year}-12-31",
        "open_access.is_oa:true",
    ]
    params = {
        "search": query,
        "filter": ",".join(filters),
        "per_page": min(200, limit),
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", []):
        year = item.get("publication_year")
        authors = []
        for a in item.get("authorships", []):
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        primary = item.get("primary_location") or {}
        full_text_url = primary.get("pdf_url") or primary.get("landing_page_url")
        journal = (item.get("host_venue") or {}).get("display_name")
        results.append(
            Paper(
                title=item.get("title") or "",
                abstract=(item.get("abstract") or ""),
                year=int(year) if year else None,
                authors=authors,
                journal=journal,
                doi=(item.get("doi") or "").replace("https://doi.org/", ""),
                url=primary.get("landing_page_url"),
                full_text_url=full_text_url,
                source="OpenAlex",
            )
        )
    return results


def doaj_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    url = f"https://doaj.org/api/v2/search/articles/{requests.utils.quote(query)}"
    params = {"page": 1, "pageSize": limit}
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", []):
        bib = item.get("bibjson", {})
        year = bib.get("year")
        if year and not (start_year <= int(year) <= end_year):
            continue
        authors = [a.get("name") for a in bib.get("author", []) if a.get("name")]
        links = bib.get("link", [])
        full_text_url = None
        for link in links:
            if link.get("type") in ("fulltext", "pdf"):
                full_text_url = link.get("url")
                break
        doi = None
        for ident in bib.get("identifier", []):
            if ident.get("type") == "doi":
                doi = ident.get("id")
                break
        results.append(
            Paper(
                title=bib.get("title") or "",
                abstract=bib.get("abstract") or "",
                year=int(year) if year else None,
                authors=authors,
                journal=bib.get("journal", {}).get("title"),
                doi=doi,
                url=bib.get("link", [{}])[0].get("url"),
                full_text_url=full_text_url,
                source="DOAJ",
                volume=bib.get("journal", {}).get("volume"),
                issue=bib.get("journal", {}).get("number"),
                pages=bib.get("journal", {}).get("pages"),
            )
        )
    return results


def semantic_scholar_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,venue,journal,openAccessPdf,url,doi,publicationTypes,volume,issue,pages",
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("data", []):
        year = item.get("year")
        if year and not (start_year <= int(year) <= end_year):
            continue
        authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
        pdf = item.get("openAccessPdf") or {}
        results.append(
            Paper(
                title=item.get("title") or "",
                abstract=item.get("abstract") or "",
                year=int(year) if year else None,
                authors=authors,
                journal=(item.get("journal") or {}).get("name") or item.get("venue"),
                doi=item.get("doi"),
                url=item.get("url"),
                full_text_url=pdf.get("url"),
                source="Semantic Scholar",
                volume=item.get("volume"),
                issue=item.get("issue"),
                pages=item.get("pages"),
            )
        )
    return results


def crossref_search(query: str, start_year: int, end_year: int, limit: int) -> List[Paper]:
    url = "https://api.crossref.org/works"
    filters = [
        f"from-pub-date:{start_year}-01-01",
        f"until-pub-date:{end_year}-12-31",
        "type:journal-article",
        "license.url:*",
    ]
    params = {
        "query.bibliographic": query,
        "filter": ",".join(filters),
        "rows": limit,
        "mailto": "example@example.org",
    }
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("message", {}).get("items", []):
        issued = item.get("issued", {}).get("date-parts", [[None]])[0]
        year = issued[0] if issued and issued[0] else None
        authors = []
        for a in item.get("author", []):
            name = " ".join([p for p in [a.get("given"), a.get("family")] if p])
            if name:
                authors.append(name)
        links = item.get("link", [])
        full_text_url = None
        for link in links:
            if link.get("content-type") in ("application/pdf", "text/html"):
                full_text_url = link.get("URL")
                break
        results.append(
            Paper(
                title=(item.get("title") or [""])[0],
                abstract=re.sub(r"<[^>]+>", "", item.get("abstract") or ""),
                year=int(year) if year else None,
                authors=authors,
                journal=(item.get("container-title") or [None])[0],
                doi=item.get("DOI"),
                url=item.get("URL"),
                full_text_url=full_text_url,
                source="Crossref",
                volume=(item.get("volume")),
                issue=(item.get("issue")),
                pages=(item.get("page")),
            )
        )
    return results


@st.cache_data(show_spinner=False)
def run_search(
    query: str,
    start_year: int,
    end_year: int,
    max_results: int,
    sources: List[str],
) -> List[Paper]:
    per_source = max(5, math.ceil(max_results / max(1, len(sources))))
    funcs = {
        "Europe PMC": europe_pmc_search,
        "DOAJ": doaj_search,
        "Semantic Scholar": semantic_scholar_search,
        "Crossref": crossref_search,
        "OpenAlex (OA)": openalex_search,
        "arXiv": arxiv_search,
        "bioRxiv": biorxiv_search,
    }
    results: List[Paper] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as ex:
        futures = [
            ex.submit(funcs[name], query, start_year, end_year, per_source)
            for name in sources
            if name in funcs
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.extend(fut.result())
            except Exception:
                continue
    return results


def build_markdown(papers: List[Paper], query: str, start_year: int, end_year: int) -> str:
    lines = [
        f"# Open Access Literature Review",
        f"Query: {query}",
        f"Years: {start_year}–{end_year}",
        "",
    ]
    for i, p in enumerate(papers, 1):
        lines.append(f"## {i}. {p.title}")
        lines.append(f"Relevance: {p.relevance:.3f} | Year: {p.year} | Source: {p.source}")
        if p.full_text_url:
            lines.append(f"Full text: {p.full_text_url}")
        if p.url:
            lines.append(f"Landing page: {p.url}")
        summary = summarize_abstract(p.abstract)
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append(f"APA: {apa_citation(p)}")
        lines.append("")
    return "\n".join(lines)


def apply_journal_filter(papers: List[Paper], journal_filter: str) -> List[Paper]:
    terms = [t.strip().lower() for t in (journal_filter or "").split(",") if t.strip()]
    if not terms:
        return papers
    out = []
    for p in papers:
        name = (p.journal or "").lower()
        if any(term in name for term in terms):
            out.append(p)
    return out


def main() -> None:
    st.set_page_config(page_title="Open Access Literature Agent", layout="wide")
    st.title("Open Access Literature Search & Review")

    with st.sidebar:
        st.header("Search Inputs")
        topic = st.text_input("Topic", "")
        organisms = st.text_input("Organism(s)", "")
        current_year = dt.date.today().year
        start_year = st.number_input("Start year", min_value=1900, max_value=current_year, value=2015)
        end_year = st.number_input("End year", min_value=1900, max_value=current_year, value=current_year)
        max_results = st.number_input("Max results", min_value=5, max_value=500, value=50)
        relevance_threshold = st.slider("Relevance threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        journal_filter = st.text_input("Journal/venue filter (optional, comma-separated)", "Nature, Science")
        sources = st.multiselect(
            "Sources",
            ["Europe PMC", "DOAJ", "Semantic Scholar", "Crossref", "OpenAlex (OA)", "arXiv", "bioRxiv"],
            default=["Europe PMC", "DOAJ", "Semantic Scholar", "Crossref", "OpenAlex (OA)", "arXiv", "bioRxiv"],
        )
        run = st.button("Run search")

    if not run:
        st.info("Enter inputs and click Run search.")
        return

    if not topic.strip():
        st.error("Topic is required.")
        return

    query = topic
    if organisms.strip():
        query = f"{topic} AND ({organisms})"

    with st.spinner("Searching open access sources..."):
        papers = run_search(query, int(start_year), int(end_year), int(max_results), sources)

    if not papers:
        st.warning("No results found.")
        return

    tfidf_relevance(query, papers)
    papers = apply_journal_filter(papers, journal_filter)
    papers = [p for p in papers if p.relevance >= relevance_threshold]
    papers.sort(key=lambda p: (p.relevance, p.year or 0), reverse=True)
    papers = papers[: int(max_results)]

    st.subheader(f"Results ({len(papers)})")
    for p in papers:
        with st.container():
            st.markdown(f"### {p.title}")
            meta = f"Relevance: {p.relevance:.3f} | Year: {p.year} | Source: {p.source}"
            st.write(meta)
            summary = summarize_abstract(p.abstract)
            if summary:
                st.write(summary)
            if p.full_text_url:
                st.markdown(f"Full text: {p.full_text_url}")
            if p.url:
                st.markdown(f"Landing page: {p.url}")
            st.write("APA:")
            st.write(apa_citation(p))
            st.divider()

    md = build_markdown(papers, query, int(start_year), int(end_year))
    js = json.dumps([p.to_dict() for p in papers], indent=2)

    st.download_button("Download Markdown report", md, file_name="literature_report.md")
    st.download_button("Download JSON", js, file_name="literature_results.json")


if __name__ == "__main__":
    main()
