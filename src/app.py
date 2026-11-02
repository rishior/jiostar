"""
Streamlit web application for the Contract Intelligence Assistant.

Provides an enterprise interface for querying legal agreements,
retrieving relevant provisions with exact character offset citations,
and enforcing strict abstention policies.
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.config import PipelineConfig, CONFIGS_DIR, DATA_DIR
from src.retrieval.pipeline import RetrievalPipeline, RetrievalResult, strip_contract_prefix
from src.data.loader import load_benchmark


@st.cache_resource
def load_pipeline():
    """Load the retrieval pipeline cached across reruns."""
    cfg_path = CONFIGS_DIR / "final_v2.yaml"
    if not cfg_path.exists():
        cfg_path = CONFIGS_DIR / "final.yaml"
    if cfg_path.exists():
        cfg = PipelineConfig.load(cfg_path)
    else:
        cfg = PipelineConfig()

    pipeline = RetrievalPipeline(cfg)
    queries = load_benchmark()
    pipeline.build_index(queries)
    return pipeline


def clean_display(text: str) -> str:
    """Sanitize strings for display to remove hyphens and maintain clean typography."""
    if not text:
        return ""
    return str(text).replace("-", " ")


def render_ui():
    st.set_page_config(
        page_title="Contract Intelligence Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Enterprise styling
    st.markdown(
        """
        <style>
        .main {
            padding-top: 1rem;
        }
        .app_header {
            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
            padding-bottom: 1.2rem;
            margin-bottom: 1.5rem;
        }
        .app_title {
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            color: inherit;
        }
        .app_subtitle {
            font-size: 0.95rem;
            color: #888888;
            margin-top: 0.3rem;
            margin-bottom: 0;
        }
        .metric_badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        .card_container {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 6px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            background: rgba(128, 128, 128, 0.03);
        }
        .card_header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(128, 128, 128, 0.15);
            padding-bottom: 0.6rem;
            margin-bottom: 0.8rem;
        }
        .provision_rank {
            font-size: 0.85rem;
            font-weight: 700;
            color: #3b82f6;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .score_pill {
            font-size: 0.8rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 3px;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .meta_row {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            font-size: 0.82rem;
            color: #888888;
            margin-bottom: 0.8rem;
        }
        .meta_label {
            font-weight: 600;
            color: inherit;
        }
        .clause_box {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.88rem;
            line-height: 1.6;
            padding: 1rem;
            border-radius: 4px;
            background: rgba(0, 0, 0, 0.25);
            border-left: 3px solid #3b82f6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .abstain_box {
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 6px;
            padding: 1.5rem;
            background: rgba(239, 68, 68, 0.08);
            margin-top: 1rem;
        }
        .abstain_title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f87171;
            margin-bottom: 0.4rem;
        }
        .abstain_desc {
            font-size: 0.9rem;
            color: #cccccc;
            margin-bottom: 0;
            line-height: 1.5;
        }
        .sidebar_section {
            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }
        .sidebar_title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888888;
            margin-bottom: 0.6rem;
        }
        .stat_item {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            font-size: 0.85rem;
        }
        .stat_key {
            color: #888888;
        }
        .stat_val {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    pipeline = load_pipeline()

    # Application Header
    st.markdown(
        """
        <div class="app_header">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <h1 class="app_title">Contract Intelligence Assistant</h1>
                    <p class="app_subtitle">Enterprise legal clause search with character offset citation and verified abstention</p>
                </div>
                <div>
                    <span class="metric_badge">Index Online</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state keys
    if "input_query" not in st.session_state:
        st.session_state["input_query"] = ""
    if "auto_search" not in st.session_state:
        st.session_state["auto_search"] = False
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "last_router_status" not in st.session_state:
        st.session_state["last_router_status"] = None
    if "last_stripped_query" not in st.session_state:
        st.session_state["last_stripped_query"] = None

    # Sidebar specifications and diagnostics
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar_section">
                <div class="sidebar_title">System Architecture</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cfg_name = clean_display(pipeline.config.name)
        embed_name = clean_display(pipeline.config.embedding.model_name)
        chunk_size = pipeline.config.chunking.chunk_size
        chunk_overlap = pipeline.config.chunking.overlap
        heading_aware = getattr(pipeline.config.chunking, "heading_aware", False)
        strip_prefix = getattr(pipeline.config.retrieval, "strip_query_prefix", False)
        use_routing = pipeline.config.retrieval.use_routing
        score_thresh = pipeline.config.abstention.score_threshold
        vectors_count = pipeline.index.size if pipeline.index else 0

        st.markdown(
            f"""
            <div class="stat_item"><span class="stat_key">Pipeline Version</span><span class="stat_val">v2 Production</span></div>
            <div class="stat_item"><span class="stat_key">Configuration</span><span class="stat_val">{cfg_name}</span></div>
            <div class="stat_item"><span class="stat_key">Embedding Model</span><span class="stat_val">{embed_name}</span></div>
            <div class="stat_item"><span class="stat_key">Chunk Size</span><span class="stat_val">{chunk_size} chars</span></div>
            <div class="stat_item"><span class="stat_key">Chunk Overlap</span><span class="stat_val">{chunk_overlap} chars</span></div>
            <div class="stat_item"><span class="stat_key">Heading Aware</span><span class="stat_val">{'Enabled' if heading_aware else 'Disabled'}</span></div>
            <div class="stat_item"><span class="stat_key">Prefix Stripping</span><span class="stat_val">{'Enabled' if strip_prefix else 'Disabled'}</span></div>
            <div class="stat_item"><span class="stat_key">Document Routing</span><span class="stat_val">{'Enabled' if use_routing else 'Disabled'}</span></div>
            <div class="stat_item"><span class="stat_key">Indexed Vectors</span><span class="stat_val">{vectors_count}</span></div>
            <div class="stat_item"><span class="stat_key">Score Threshold</span><span class="stat_val">{score_thresh}</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar_section" style="margin-top: 1.5rem;">
                <div class="sidebar_title">Benchmark Test Queries</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        example_queries = [
            ("Sublicensing Permissions", "Does this agreement allow sublicensing to third parties?"),
            ("Exclusivity Duration", "When does the exclusivity period end?"),
            ("Intellectual Property Ownership", "Who owns the IP in derivative works?"),
            ("Termination Conditions", "What are the termination conditions?"),
            ("Indemnification Scope", "Is there an indemnification clause?"),
            ("Absent Clause (In Corpus)", "What are the cryptocurrency dividend distribution rules?"),
            ("Absent Contract (External)", "Consider the Netflix Agreement; does it permit brand sharing?"),
        ]

        for label, q_text in example_queries:
            if st.button(label, key=f"btn_{abs(hash(label))}", use_container_width=True):
                st.session_state["input_query"] = q_text
                st.session_state["auto_search"] = True
                st.rerun()

    # Query Input Section
    col_input, col_k = st.columns([5, 1])
    with col_input:
        query_text = st.text_area(
            label="Search Query or Legal Question",
            value=st.session_state["input_query"],
            placeholder="Enter question or paste benchmark clause query",
            height=85,
        )
    with col_k:
        top_k = st.number_input("Top Results", min_value=1, max_value=20, value=5)

    col_btn_search, col_btn_clear, col_fill = st.columns([2, 1, 5])
    with col_btn_search:
        execute = st.button("Search Corpus", type="primary", use_container_width=True)
    with col_btn_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state["input_query"] = ""
            st.session_state["last_results"] = None
            st.session_state["last_router_status"] = None
            st.session_state["last_stripped_query"] = None
            st.rerun()

    # Trigger retrieval if search clicked or auto search requested
    should_run = execute or st.session_state.pop("auto_search", False)

    if should_run and query_text.strip():
        search_query = query_text.strip()
        pipeline.config.retrieval.top_k = top_k

        # Router diagnostics
        router_status = None
        if pipeline.router and pipeline.config.retrieval.use_routing:
            target_f, conf, named = pipeline.router.route(search_query)
            if named:
                if target_f:
                    clean_target = clean_display(Path(target_f).stem)
                    router_status = f"Agreement identified in corpus ({clean_target})"
                else:
                    router_status = "Named agreement is not present in the indexed corpus"
            else:
                router_status = "General query (global corpus search across all contracts)"
        st.session_state["last_router_status"] = router_status

        # Query prefix stripping check
        stripped_info = None
        if getattr(pipeline.config.retrieval, "strip_query_prefix", False):
            substantive_q = strip_contract_prefix(search_query)
            if substantive_q != search_query:
                stripped_info = clean_display(substantive_q)
        st.session_state["last_stripped_query"] = stripped_info

        with st.spinner("Executing retrieval pipeline..."):
            results = pipeline.query(search_query, top_k=top_k)
            st.session_state["last_results"] = results

    # Render results if available
    if st.session_state["last_router_status"]:
        st.info(f"Router Status: {st.session_state['last_router_status']}")

    if st.session_state["last_stripped_query"]:
        st.caption(f"Stripped Query: {st.session_state['last_stripped_query']}")

    if st.session_state["last_results"] is not None:
        results = st.session_state["last_results"]
        if not results:
            st.markdown(
                """
                <div class="abstain_box">
                    <div class="abstain_title">System Decision: Abstain from Retrieval</div>
                    <p class="abstain_desc">
                        No candidate passage in the corpus exceeded the relevance threshold of 0.30,
                        or the named agreement is absent from the indexed repository.
                        In accordance with precision guarantees, the system abstains from returning unverified text.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="margin: 1.2rem 0 0.8rem 0; font-size: 0.95rem; font-weight: 600;">
                    Retrieved {len(results)} verified provisions ordered by relevance score
                </div>
                """,
                unsafe_allow_html=True,
            )

            for i, r in enumerate(results, 1):
                clean_doc = clean_display(Path(r.file_path).stem)
                clean_path = clean_display(r.file_path)
                span_len = r.end - r.start

                st.markdown(
                    f"""
                    <div class="card_container">
                        <div class="card_header">
                            <span class="provision_rank">Provision {i} | {clean_doc}</span>
                            <span class="score_pill">Score {r.score:.4f}</span>
                        </div>
                        <div class="meta_row">
                            <div><span class="meta_label">Document:</span> {clean_path}</div>
                            <div><span class="meta_label">Offsets:</span> [{r.start} to {r.end}]</div>
                            <div><span class="meta_label">Length:</span> {span_len} characters</div>
                            <div><span class="meta_label">Chunk ID:</span> {r.chunk_id}</div>
                        </div>
                        <div class="clause_box">{r.text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def main():
    render_ui()


if __name__ == "__main__":
    main()
