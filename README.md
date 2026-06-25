# Carbon Stock Modeling with Multisensor Remote Sensing and AI: A Bibliometric Analysis

This repository contains the data, processing pipeline, and interactive dashboard for a bibliometric study of the scientific literature on carbon stock modeling using multisensor remote sensing and artificial intelligence (2015–2025). The dashboard is deployed on Streamlit Community Cloud from this repository.

## 1. Data sources and search strategy

Bibliographic records were retrieved from the Web of Science Core Collection and Scopus, covering the publication period 2015–2025. The initial query combined remote sensing and artificial intelligence terms with vegetation/forest/agriculture-related terms, which also captured a substantial number of precision-agriculture studies (crop yield prediction, soil and nutrient mapping) that apply the same methodological toolkit without addressing carbon stock.

To keep the corpus aligned with the study's scope, records were screened for an explicit, carbon/biomass-specific term in the title, abstract, or author/index keywords (e.g., *carbon stock*, *carbon sequestration*, *carbon density*, *soil organic carbon*, *aboveground biomass*, *forest carbon*, *carbon cycle*, *carbon flux*). This screening step removed records that, despite matching the search syntax, were thematically unrelated to carbon stock estimation. Several of the most-cited records before screening, for instance, were general remote-sensing/machine-learning methodology papers or crop-monitoring studies with no carbon stock component. The corpus was reduced from 3,689 to 1,755 records after screening. This illustrates a recurring difficulty in bibliometric searches at the intersection of methodology-driven fields (remote sensing, AI) and an application domain (carbon stock): a syntactically precise query can still return a topically heterogeneous corpus, and keyword-based post-hoc screening, while necessary, cannot achieve perfect precision without manual curation.

## 2. Corpus construction

1. Conversion. Web of Science exports (.xls) were converted to the tagged-plaintext format read by `bibliometrix::convert2df()`; Scopus exports were processed directly from CSV.
2. Merging and deduplication. Records from both sources were combined and deduplicated, first by normalized DOI, then by normalized title and publication year for records without a DOI. When the same article was indexed in both databases, the record with a non-empty cited-reference list was retained preferentially, which materially improved local-citation coverage (see Section 4).
3. Thematic screening. As described in Section 1.
4. Field enrichment. Author country (`AU_CO`) was derived from affiliation strings; a merged keyword field (`KW_Merged`) combines author keywords and Keywords Plus, deduplicated.

The final corpus comprises 1,755 unique documents (2015–2025): 138 from Web of Science and 1,617 from Scopus after deduplication, 108 with at least one Brazilian-affiliated author.

## 3. Analytical methods

Two complementary layers of analysis were applied to the corpus.

Classic bibliometric indicators (R, `bibliometrix` 5.4.0): annual scientific production, Bradford's Law, Lotka's Law, thematic mapping (strategic diagram), trend topics, cumulative keyword growth, h-index ranking among the most productive authors and journals, and local/global citation analysis. These indicators are computed once over the full corpus and exposed in the dashboard's "Classic Indicators" and "Networks" tabs; they are not recomputed against the sidebar filters.

Complementary text-mining layer (Python): detection of AI/ML techniques, sensor and data types, drone/UAV usage, application areas, and affiliation countries via dictionary-based, word-boundary-matched regular expressions over title, abstract, and keywords. This layer powers the dashboard's interactive tabs (Overview, Authors & Journals, AI Techniques, Data Types, Drone Analysis, Geographic Distribution, Area Focus), which recompute live against the sidebar filters (year range, study origin, document type, top-N).

Network visualizations (keyword co-occurrence, co-authorship, international collaboration, and their Brazil-only counterparts) are rendered as interactive force-directed graphs.

## 4. Known limitations

Only 90.3% of records (1,585 of 1,755) carry a complete cited-reference list, which restricts local citation and co-citation analysis to that subset. This stems from an institutional access-tier limit on Web of Science's cited-reference export (the "Cited Reference Count" field is populated, but the reference list text is not), rather than a data-processing issue; it was confirmed by re-exporting the source records and observing the same gap.

Word-boundary matching was applied throughout the text-mining layer to avoid short-acronym false positives (e.g., "ANN" matching inside "channel", "GAN" inside "organic", "ALS" inside "materials"). Categories such as AI techniques and data/sensor types are not mutually exclusive: a single document can be tagged with multiple techniques or data types, which is intentional for a co-occurrence-style analysis but should be kept in mind when reading frequency counts as if they summed to the corpus size.

## 5. Repository structure

```
streamlit_dashboard.py     Main application: sidebar filters and the 7 filter-reactive tabs
bibliometric_analysis.py   Text-mining layer (BibliometricAnalyzer class)
classic_indicators.py      Classic-indicator, network, and methodology tabs (static, full-corpus)
data/json/                 Precomputed bibliometrix/R and Python outputs consumed by classic_indicators.py
DF_COMBINADO_LIMPO.csv     Deduplicated, screened corpus (1,755 records, 63 fields)
requirements.txt           Python dependencies
runtime.txt                Pinned Python version for Streamlit Community Cloud
```

The R and Python scripts that produce `DF_COMBINADO_LIMPO.csv` and `data/json/` from the raw Web of Science/Scopus exports are maintained outside this repository, alongside the source `.xls`/`.csv` exports, for reproducibility of the full pipeline.

## 6. Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```
