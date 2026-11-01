"""
Streamlit web application for the Contract Intelligence Assistant.

Provides a simple interface for querying legal contracts and viewing
the retrieved passages with their source locations.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.config import PipelineConfig, CONFIGS_DIR, DATA_DIR
from src.retrieval.pipeline import RetrievalPipeline


@st.cache_resource
def load_pipeline():
    """Load the retrieval pipeline (cached across reruns)."""
    cfg_path = CONFIGS_DIR / "final.yaml"
    if cfg_path.exists():
        cfg = PipelineConfig.load(cfg_path)
    else:
        cfg = PipelineConfig()

    pipeline = RetrievalPipeline(cfg)
    from src.data.loader import load_benchmark
    queries = load_benchmark()
    pipeline.build_index(queries)

    return pipeline


def main():
    st.set_page_config(
        page_title="Contract Intelligence Assistant",
        page_icon="",
        layout="wide",
    )

    st.title("Contract Intelligence Assistant")
    st.markdown(
        "Search legal contracts for specific clauses. Enter a natural language "
        "question below to find the most relevant passages from the contract corpus."
    )

    pipeline = load_pipeline()

    # Query input
    st.subheader("Query")
    question = st.text_area(
        "Enter your question about the contracts:",
        placeholder="Does this agreement allow sublicensing to third parties?",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.number_input("Results to show", min_value=1, max_value=20, value=5)
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)

    if search_button and question.strip():
        pipeline.config.retrieval.top_k = top_k

        with st.spinner("Searching contracts..."):
            results = pipeline.query(question.strip())

        if not results:
            st.warning(
                "No relevant passages found. The contract corpus does not appear "
                "to contain a clause answering this question."
            )
        else:
            st.success(f"Found {len(results)} relevant passage(s)")

            for i, r in enumerate(results, 1):
                with st.expander(
                    f"Result {i}: {Path(r.file_path).stem} (score: {r.score:.4f})",
                    expanded=(i <= 3),
                ):
                    st.markdown(f"**Document:** `{r.file_path}`")
                    st.markdown(f"**Character offsets:** [{r.start}:{r.end}]")
                    st.markdown(f"**Relevance score:** {r.score:.4f}")
                    st.markdown("**Passage text:**")
                    st.text(r.text)

    # Sidebar with system info
    with st.sidebar:
        st.subheader("System Information")
        st.markdown(f"**Configuration:** {pipeline.config.name}")
        st.markdown(f"**Embedding model:** {pipeline.config.embedding.model_name}")
        st.markdown(f"**Chunk size:** {pipeline.config.chunking.chunk_size}")
        st.markdown(f"**Chunk overlap:** {pipeline.config.chunking.overlap}")
        if pipeline.index:
            st.markdown(f"**Index size:** {pipeline.index.size} vectors")
        st.markdown(f"**BM25 hybrid:** {'Enabled' if pipeline.config.retrieval.use_bm25 else 'Disabled'}")
        st.markdown(f"**Abstention threshold:** {pipeline.config.abstention.score_threshold}")

        st.subheader("Example Questions")
        examples = [
            "Does this agreement allow sublicensing to third parties?",
            "When does the exclusivity period end?",
            "Who owns the IP in derivative works?",
            "What are the termination conditions?",
            "Is there an indemnification clause?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                st.session_state["question"] = ex
                st.rerun()


if __name__ == "__main__":
    main()
