#!/usr/bin/env python3
"""
Classic bibliometrix indicators and citation/keyword networks — v2 addition.

These tabs are computed once (offline, in R with bibliometrix) over the
FULL 2015-2025 corpus and shipped as static JSON files in data/json/. They
are NOT affected by the sidebar filters (year range / study type), unlike
the original tabs in streamlit_dashboard.py, which remain untouched.

Kept in its own module on purpose: nothing here modifies the existing
analysis code, it only adds new read-only views on top of it.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyvis.network import Network

DATA_DIR = Path(__file__).parent / "data" / "json"


@st.cache_data
def _load(name):
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _df(name):
    data = _load(name)
    if data is None:
        return pd.DataFrame()
    return pd.DataFrame(data)


@st.cache_data
def _pyvis_network_html(edges, node_color="#2E8B57", edge_color="#4682B4", height_px=620):
    """Interactive force-directed network (draggable nodes, live physics) via
    vis.js, matching the behavior of the original D3.js dashboard - a static
    Plotly scatter can't drag nodes, vis.js/pyvis can."""
    if not edges:
        return None

    degree = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + e.get("weight", 1)
        degree[e["target"]] = degree.get(e["target"], 0) + e.get("weight", 1)
    max_degree = max(degree.values())

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff",
                   font_color="#222222", cdn_resources="remote")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=110,
                   spring_strength=0.04, damping=0.9)

    added = set()
    for e in edges:
        for node in (e["source"], e["target"]):
            if node not in added:
                size = 10 + 30 * (degree[node] / max_degree)
                net.add_node(node, label=node, value=size, color=node_color,
                             title=f"{node} (degree {degree[node]})")
                added.add(node)
        net.add_edge(e["source"], e["target"], value=e.get("weight", 1), color=edge_color)

    html = net.generate_html()
    # vis.js leaves the camera at its initial position, which is rarely
    # centered on the node cluster once Barnes-Hut physics settles. Fitting
    # immediately after construction can run before the iframe container has
    # a real size, producing a degenerate viewport that a later fit() doesn't
    # always correct - so the only fit calls are after stabilization, plus a
    # delayed fallback in case that event never fires (e.g. dense graphs that
    # never fully settle).
    html = html.replace(
        "network = new vis.Network(container, data, options);",
        "network = new vis.Network(container, data, options);\n"
        "              network.once('stabilizationIterationsDone', function () { network.fit({animation: false}); });\n"
        "              setTimeout(function () { network.fit({animation: false}); }, 800);",
    )
    return html


def render_classic_indicators_tab():
    st.markdown('<h2 class="sub-header">📚 Classic Bibliometric Indicators (bibliometrix)</h2>', unsafe_allow_html=True)
    st.caption(
        "Computed once, offline, with R/bibliometrix over the full 2015-2025 corpus (1,755 documents). "
        "Unlike the tabs above, this section does not react to the sidebar filters."
    )

    main_info = _df("main_information")
    if not main_info.empty:
        info = {row["Description"]: row["Results"] for _, row in main_info.iterrows() if row["Description"]}
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents", info.get("Documents", "—"))
        col2.metric("Sources (journals, etc.)", info.get("Sources (Journals, Books, etc)", "—"))
        col3.metric("Annual growth rate %", info.get("Annual Growth Rate %", "—"))
        col4.metric("Avg. citations/doc", info.get("Average citations per doc", "—"))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Annual Scientific Production")
        ap = _df("annual_production")
        if not ap.empty:
            fig = px.bar(ap, x="Year", y="Articles", color="Articles", color_continuous_scale="Greens")
            fig.update_layout(template="plotly_white", height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            growth = _load("annual_growth_rate")
            if growth is not None:
                st.caption(f"Compound annual growth rate: **{growth}%**")

    with col2:
        st.markdown("### Bradford's Law (Journal Core)")
        bf = _df("bradford_law")
        if not bf.empty:
            top = bf.head(25)
            fig = px.bar(top, x="SO", y="Freq", color="Freq", color_continuous_scale="Blues")
            fig.update_layout(template="plotly_white", height=420, xaxis_tickangle=-45, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            n_core = (bf["Zone"] == "Zone 1").sum()
            st.caption(f"{n_core} journals form the core (Zone 1) of highest productivity, per Bradford's Law.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Lotka's Law (Author Productivity)")
        lotka = _load("lotka_law")
        if lotka:
            ap2 = pd.DataFrame(lotka["author_prod"])
            beta, c = lotka["beta"], lotka["c_const"]
            xs = ap2["Documents written"]
            ys_theo = c / xs.pow(beta)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ap2["Proportion of Authors"], mode="markers",
                                      name="Observed", marker=dict(color="#2E8B57", size=9)))
            fig.add_trace(go.Scatter(x=xs, y=ys_theo, mode="lines", name="Theoretical (Lotka)",
                                      line=dict(color="#4682B4", dash="dash")))
            fig.update_layout(template="plotly_white", height=420, xaxis_type="log", yaxis_type="log",
                               xaxis_title="Documents published", yaxis_title="Proportion of authors")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"β = {beta} · R² = {lotka['r2']} — the closer to 2, the more concentrated "
                       "production is among few authors.")

    with col2:
        st.markdown("### Most Productive Countries")
        cp = _df("most_prod_countries")
        if not cp.empty:
            top = cp.head(15).iloc[::-1]
            fig = px.bar(top, x="Articles", y="Country", orientation="h", color="Articles",
                         color_continuous_scale="Greens")
            fig.update_layout(template="plotly_white", height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### h-index Among the 50 Most Productive Journals")
        st.dataframe(_df("source_hindex")[["Element", "h_index", "g_index", "TC", "NP"]]
                     .rename(columns={"Element": "Journal", "TC": "Citations", "NP": "Docs"}),
                     use_container_width=True, hide_index=True)
        st.caption("h-index computed only for the 50 journals with the most published documents — "
                   "not a corpus-wide ranking, since computing it for all 1,056 journals is "
                   "computationally impractical.")
    with col2:
        st.markdown("### h-index Among the 50 Most Productive Authors")
        st.dataframe(_df("author_hindex")[["Element", "h_index", "g_index", "TC", "NP"]]
                     .rename(columns={"Element": "Author", "TC": "Citations", "NP": "Docs"}),
                     use_container_width=True, hide_index=True)
        st.caption("h-index computed only for the 50 authors with the most published documents — "
                   "a less prolific but highly-cited author could have a higher true h-index without "
                   "appearing here.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Most Globally Cited Documents")
        mc = _df("most_cited_papers")
        if not mc.empty:
            st.dataframe(mc[["Paper", "TC", "TCperYear"]].rename(
                columns={"Paper": "Document", "TC": "Citations", "TCperYear": "Cit./Year"}),
                use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### Most Cited References (Local Citations)")
        lc = _df("most_local_cited_documents")
        if not lc.empty:
            st.dataframe(lc[["Paper", "LCS", "GCS"]].rename(
                columns={"Paper": "Document", "LCS": "Local Cit.", "GCS": "Global Cit."}),
                use_container_width=True, hide_index=True)
        st.caption("Computed on the ~25% of the corpus with complete cited-reference lists "
                   "(mostly Scopus records — this WoS export did not include the Cited References field).")

    st.markdown("---")
    st.markdown("### Strategic Diagram (Thematic Map)")
    clusters = _df("thematic_map_clusters")
    if not clusters.empty:
        fig = px.scatter(
            clusters, x="centrality", y="density", size="freq", color="name",
            text="name", size_max=60,
            labels={"centrality": "Centrality (relevance)", "density": "Density (development)"},
        )
        fig.update_traces(textposition="top center")
        mean_x, mean_y = clusters["centrality"].mean(), clusters["density"].mean()
        fig.add_vline(x=mean_x, line_dash="dot", line_color="gray")
        fig.add_hline(y=mean_y, line_dash="dot", line_color="gray")
        fig.update_layout(template="plotly_white", height=520, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Axes: centrality (relevance) vs. density (internal theme development). "
                   "Bubble size = cluster frequency.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Trend Topics")
        tt = _df("trend_topics")
        if not tt.empty:
            tt = tt.sort_values("year_med")
            fig = go.Figure(go.Scatter(
                x=tt["year_med"], y=tt["item"], mode="markers",
                marker=dict(size=8 + tt["freq"] ** 0.5 * 2, color="#2E8B57"),
                error_x=dict(type="data", symmetric=False,
                             array=tt["year_q3"] - tt["year_med"],
                             arrayminus=tt["year_med"] - tt["year_q1"], color="#4682B4"),
            ))
            fig.update_layout(template="plotly_white", height=max(420, len(tt) * 18),
                               margin=dict(l=160))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Cumulative Growth of Leading Keywords")
        kg = _df("keyword_growth")
        if not kg.empty:
            kg_long = kg.melt(id_vars="Year", var_name="Keyword", value_name="Occurrences")
            fig = px.line(kg_long, x="Year", y="Occurrences", color="Keyword", markers=True)
            fig.update_layout(template="plotly_white", height=480)
            st.plotly_chart(fig, use_container_width=True)


def render_networks_tab():
    import streamlit.components.v1 as components

    st.markdown('<h2 class="sub-header">🕸️ Citation, Keyword & Collaboration Networks</h2>', unsafe_allow_html=True)
    st.caption(
        "Interactive force-directed networks (vis.js) computed over the full 2015-2025 corpus. "
        "Drag a node to reposition it, scroll/pinch to zoom, hover for its degree."
    )

    st.markdown("### Keyword Co-occurrence Network")
    kw_edges = _load("network_keyword_cooc")
    html = _pyvis_network_html(kw_edges, node_color="#2E8B57", edge_color="#4682B4")
    if html:
        components.html(html, height=640, scrolling=False)
    else:
        st.info("No keyword network data available.")

    st.markdown("---")
    st.markdown("### Co-authorship Network")
    au_edges = _load("network_author_collab")
    html = _pyvis_network_html(au_edges, node_color="#4682B4", edge_color="#2E8B57")
    if html:
        components.html(html, height=640, scrolling=False)
    else:
        st.info("No co-authorship network data available.")

    st.markdown("---")
    st.markdown("### International Collaboration Network")
    co_edges = _load("network_country_collab_py")
    html = _pyvis_network_html(co_edges, node_color="#E07A5F", edge_color="#9B5DE5")
    if html:
        components.html(html, height=640, scrolling=False)
    else:
        st.info("No country collaboration network data available.")
    ic = _load("international_collab_stats")
    if ic:
        st.caption(f"{ic['international_collab_papers']:,} articles ({ic['pct_international']}%) "
                   "have co-authorship across institutions from different countries.")

    st.markdown("---")
    st.markdown("## 🇧🇷 Brazil-Only Networks")
    st.caption(
        "Same network types, recomputed only on the 141 documents with a Brazilian affiliation "
        "(smaller node/edge limits than the full-corpus networks above, since the subset is much "
        "smaller). These are static like the rest of this tab and don't follow the sidebar filters."
    )

    st.markdown("### Keyword Co-occurrence Network (Brazil)")
    kw_edges_br = _load("network_keyword_cooc_brazil")
    html = _pyvis_network_html(kw_edges_br, node_color="#2E8B57", edge_color="#4682B4", height_px=560)
    if html:
        components.html(html, height=580, scrolling=False)
    else:
        st.info("No keyword network data available for Brazil.")

    st.markdown("---")
    st.markdown("### Co-authorship Network (Brazil)")
    au_edges_br = _load("network_author_collab_brazil")
    html = _pyvis_network_html(au_edges_br, node_color="#4682B4", edge_color="#2E8B57", height_px=560)
    if html:
        components.html(html, height=580, scrolling=False)
    else:
        st.info("No co-authorship network data available for Brazil.")

    st.markdown("---")
    st.markdown("### Brazil's International Partners")
    partners = _load("brazil_top_partners") or []
    partner_edges = [{"source": "BRAZIL", "target": p["Country"], "weight": p["Collaborations"]}
                      for p in partners]
    html = _pyvis_network_html(partner_edges, node_color="#E07A5F", edge_color="#9B5DE5", height_px=560)
    if html:
        components.html(html, height=580, scrolling=False)
    else:
        st.info("No Brazil partner-country data available.")


def render_methodology_tab():
    st.markdown('<h2 class="sub-header">🧪 Methodology & Technical Notes</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Corpus Fact Sheet (bibliometrix)")
        rows = _load("main_information") or []
        html = ['<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">']
        for row in rows:
            desc, val = row.get("Description", ""), row.get("Results", "")
            if not desc:
                continue
            if val == "":
                html.append(
                    '<tr style="background:#2E8B57;">'
                    f'<th colspan="2" style="text-align:left; padding:7px 10px; color:white;">{desc}</th></tr>'
                )
            else:
                html.append(
                    '<tr>'
                    f'<td style="padding:4px 10px; border-bottom:1px solid #eee;">{desc}</td>'
                    f'<td style="padding:4px 10px; border-bottom:1px solid #eee; text-align:right; font-weight:600;">{val}</td>'
                    '</tr>'
                )
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)

    stats = _load("summary_stats") or {}

    with col2:
        st.markdown("### How This Analysis Was Built")
        st.markdown(
            "1. **Collection:** full-record exports from Web of Science (.xls) and Scopus (.csv) "
            "covering 2015–2025.\n"
            "2. **Conversion:** WoS converted to tagged-plaintext and processed with "
            "`bibliometrix::convert2df()`; Scopus processed directly via CSV.\n"
            "3. **Merging & deduplication:** the two sources are combined and deduplicated by "
            "normalized DOI, falling back to normalized title + year when no DOI is present. When "
            "the same article exists in both sources, the record with a non-empty cited-reference "
            "list is kept preferentially.\n"
            "4. **Thematic scope screening:** the initial query returned a substantial share of "
            "precision-agriculture records (crop yield prediction, soil/nutrient mapping) that share "
            "the same remote-sensing/AI methodology but do not address carbon stock. Records were "
            "screened by requiring an explicit carbon/biomass-specific term (e.g. carbon stock, "
            "carbon sequestration, soil organic carbon, aboveground biomass) in the title, abstract, "
            "or keywords, narrowing the corpus from 3,689 to **1,755** documents.\n"
            "5. **Classic indicators:** `biblioAnalysis`, Bradford's Law, Lotka's Law, thematic map, "
            "co-occurrence/co-authorship networks, h-index, local/global citations — computed with "
            "R/bibliometrix over the full corpus.\n"
            "6. **Complementary layer:** detection of AI techniques, sensor/data types, drone use, "
            "application areas and affiliation countries via word-boundary regular expressions "
            "(reduces false positives from short acronyms, e.g. \"ANN\" inside \"channel\", \"GAN\" "
            "inside \"organic\").\n"
            f"7. **Known limitation:** only {stats.get('pct_with_cr', '~25')}% of records have a "
            "complete cited-reference list, which restricts local citation/co-citation analysis to "
            "that subset — a Web of Science institutional access-tier limit, not a processing issue."
        )
        st.caption("Repository: github.com/RUrzeda/dashboard_carbon_stock_v2")

    st.markdown("---")
    st.markdown("### Data Sources & Reproducibility")
    st.markdown(
        "- **Sources:** Web of Science (Core Collection) + Scopus, exported directly from each platform.\n"
        f"- **Period:** {stats.get('timespan', '2015-2025')} (complete), "
        f"{stats.get('total_documents', 1755):,} unique documents after deduplication and thematic "
        f"screening ({stats.get('n_wos', 138):,} WoS + {stats.get('n_scopus', 1617):,} Scopus before merge).\n"
        "- **Pipeline:** R (bibliometrix 5.4.0, isolated in a dedicated conda environment) for classic "
        "indicators + Python for the complementary text-mining layer (AI techniques, data types, "
        "drones, geography, application areas).\n"
        "- **Tabs 1–7** above react to the sidebar filters (year range, Brazil/International, top-N) "
        "and are computed live from the deduplicated CSV.\n"
        "- **Tabs 8–10** (Classic Indicators, Networks, Methodology) are pre-computed once over the "
        "full corpus and do not change with the sidebar filters."
    )
