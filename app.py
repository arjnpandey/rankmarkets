"""
Streamlit UI for the Prediction Market Explorer.
"""

import os
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from scraper import load_cached_events, run_scraper, PARENT_CATEGORIES
from agents import (
    find_relevant_markets,
    analyze_enterprise,
    INDUSTRIES,
    GEOGRAPHIES,
    COST_INPUTS,
    REVENUE_STREAMS,
    REVENUE_TICKS,
)

st.set_page_config(page_title="Market Explorer", page_icon="📊", layout="wide")

# --- Password gate ---
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Prediction Market Explorer")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Login", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

st.title("Prediction Market Explorer")
st.caption("Search Kalshi & Polymarket with natural language")

# --- Sidebar: data status ---
with st.sidebar:
    st.header("Data")
    events, cat_index, updated_at, embeddings = load_cached_events()

    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at)
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            st.metric("Cached events", len(events))
            st.caption(f"Updated {age_hours:.1f}h ago")
            stale = age_hours > 12
        except ValueError:
            stale = True
    else:
        st.warning("No cached data yet.")
        stale = True

    if embeddings is None and events:
        st.warning("No embeddings found. Click Refresh to generate.")

    if st.button("Refresh market data", type="primary" if stale else "secondary"):
        with st.spinner("Fetching & categorizing markets..."):
            run_scraper()
        st.rerun()

    # Category breakdown
    if cat_index:
        with st.expander("Categories"):
            for cat in PARENT_CATEGORIES:
                n = len(cat_index.get(cat, []))
                if n:
                    st.text(f"{cat}: {n}")

# --- Tabs ---
tab_search, tab_enterprise = st.tabs(["Market Search", "Enterprise Profiles"])

# ===================== Tab 1: Market Search =====================
with tab_search:
    query = st.text_input(
        "What are you looking for?",
        placeholder='e.g. "AI regulation", "Federal Reserve interest rates", "Trump tariffs"',
    )

    if st.button("Search", type="primary", disabled=not query, key="search_btn"):
        if not events:
            st.error("No market data cached. Click **Refresh market data** in the sidebar first.")
        else:
            with st.spinner("Searching..."):
                try:
                    matched_cats, keywords, results = find_relevant_markets(query, events, cat_index, embeddings)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            if matched_cats:
                st.caption(f"Categories: {', '.join(matched_cats)}  |  Keywords: {', '.join(keywords)}")

            if not results:
                st.warning("No relevant markets found. Try a different query.")
            else:
                st.subheader(f"Top {len(results)} markets for \"{query}\"")
                for i, item in enumerate(results, 1):
                    event = item["event"]
                    source = event.get("source", "").upper()
                    category = event.get("parentCategory") or event.get("category", "")
                    url = event.get("url", "")
                    nested = event.get("nestedMarkets", [])

                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{i}. {event.get('eventTitle', '')}**")
                            reason = item.get("reason", "")
                            if reason:
                                st.caption(reason)
                            hedge = event.get("hedgeCase", "")
                            if hedge:
                                st.caption(f"Hedge: {hedge}")
                        with col2:
                            st.markdown(f"`{source}` · {category}")
                            if url:
                                st.markdown(f"[View on {source}]({url})")

                        if nested:
                            cols = st.columns(min(len(nested), 4))
                            for j, m in enumerate(nested[:4]):
                                with cols[j]:
                                    price = m.get("yes_price", 0)
                                    vol = m.get("volume", 0)
                                    st.metric(
                                        label=m.get("title", "")[:50],
                                        value=f"{price:.0%}",
                                        delta=f"${vol:,.0f} vol",
                                        delta_color="off",
                                    )

# ===================== Tab 2: Enterprise Profiles =====================
with tab_enterprise:
    st.subheader("Build Your Enterprise Profile")
    st.caption(
        "Describe your business below — no deep knowledge required, just pick what fits. "
        "Gemini will identify your top risks, match them to live prediction markets you could "
        "use as hedges, and suggest new markets that should exist."
    )

    with st.container(border=True):
        st.markdown(
            "**How it works:** &nbsp; Profile → Risk analysis (Flash) → Market matching (Flash) → Suggested hedges (Pro) &nbsp; · &nbsp; "
            "~7-10 seconds, 3 Gemini calls"
        )

    with st.form("enterprise_form"):
        company = st.text_input("Company name (optional)", placeholder="e.g. Acme Corp", help="Used to personalize the analysis. Leave blank for a generic industry profile.")

        col_a, col_b = st.columns(2)
        with col_a:
            industry = st.selectbox("Industry", INDUSTRIES, help="Primary sector your business operates in.")
            revenue = st.select_slider("Annual Revenue", options=REVENUE_TICKS, value="$50M", help="Approximate annual revenue — affects the scale and type of risks surfaced.")
        with col_b:
            geographies = st.multiselect("Geography", GEOGRAPHIES, default=["US — National"], help="Regions where you operate or sell. Multi-region businesses face currency, trade, and regulatory risks.")
            cost_inputs = st.multiselect("Key Cost Inputs", COST_INPUTS, help="What drives your expenses? Helps identify supply-chain and input-price risks.")

        revenue_streams = st.multiselect("Revenue Streams", REVENUE_STREAMS, help="Where does your revenue come from? B2B vs B2C, domestic vs export, etc.")

        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not events:
            st.error("No market data cached. Click **Refresh market data** in the sidebar first.")
        elif not industry or not geographies:
            st.warning("Please select at least an industry and one geography.")
        else:
            profile = {
                "company": company or "Unnamed Company",
                "industry": industry,
                "revenue": revenue,
                "geographies": geographies,
                "cost_inputs": cost_inputs,
                "revenue_streams": revenue_streams,
            }

            with st.spinner("Analyzing enterprise risks and finding relevant markets..."):
                try:
                    result = analyze_enterprise(profile, events, cat_index, embeddings)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # ── Side-by-side: Risks + Suggested Markets ────────────
            risks = result.get("risks", [])
            suggested = result.get("suggested_markets", [])

            if risks or suggested:
                col_risks, col_suggested = st.columns(2)

                with col_risks:
                    st.subheader("Identified Risks")
                    for risk in risks:
                        severity = risk.get("severity", "Medium")
                        sev_color = {"High": "red", "Medium": "orange", "Low": "green"}.get(severity, "gray")
                        with st.container(border=True):
                            st.markdown(f"**{risk.get('name', '')}**")
                            st.markdown(f":{sev_color}[{severity}] · {risk.get('category', '')}")
                            st.caption(risk.get("description", ""))

                with col_suggested:
                    st.subheader("Suggested Markets")
                    st.caption("Markets that should exist to hedge these risks.")
                    for idx, mkt in enumerate(suggested, 1):
                        with st.container(border=True):
                            st.markdown(f"**{idx}. {mkt.get('title', '')}**")
                            st.caption(f"Hedges: {mkt.get('risk_hedged', '')}")
                            st.caption(mkt.get("rationale", ""))

            # ── Existing Markets (full width below) ──────────────────
            existing = result.get("existing_markets", [])
            if existing:
                st.subheader(f"Top {len(existing)} Relevant Markets")
                for i, item in enumerate(existing, 1):
                    event = item["event"]
                    source = event.get("source", "").upper()
                    category = event.get("parentCategory") or event.get("category", "")
                    url = event.get("url", "")
                    nested = event.get("nestedMarkets", [])

                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{i}. {event.get('eventTitle', '')}**")
                            reason = item.get("reason", "")
                            if reason:
                                st.caption(reason)
                            hedge = event.get("hedgeCase", "")
                            if hedge:
                                st.caption(f"Hedge: {hedge}")
                        with col2:
                            st.markdown(f"`{source}` · {category}")
                            if url:
                                st.markdown(f"[View on {source}]({url})")

                        if nested:
                            cols = st.columns(min(len(nested), 4))
                            for j, m in enumerate(nested[:4]):
                                with cols[j]:
                                    price = m.get("yes_price", 0)
                                    vol = m.get("volume", 0)
                                    st.metric(
                                        label=m.get("title", "")[:50],
                                        value=f"{price:.0%}",
                                        delta=f"${vol:,.0f} vol",
                                        delta_color="off",
                                    )
