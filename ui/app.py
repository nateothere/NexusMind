"""NexusMind UI (Streamlit) - boots with `docker compose up`."""

import os

import httpx
import streamlit as st

API = os.environ.get("NEXUSMIND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="NexusMind", page_icon="🧠")
st.title("🧠 NexusMind")

query = st.text_area("Ask your knowledge base", height=90)
collection = st.text_input("Collection", value="default")
stream = st.checkbox("Stream answer", value=True)

if st.button("Ask", type="primary") and query:
    with st.spinner("thinking..."):
        try:
            resp = httpx.post(
                f"{API}/v1/ask",
                json={"query": query, "stream": stream, "collection": collection},
                timeout=120,
            )
            resp.raise_for_status()
            if stream:
                events = [line for line in resp.text.splitlines() if line.startswith("data: ")]
                text = "".join(
                    e[6:].strip('"') for e in events if '"delta"' in e
                )
                st.markdown(text)
            else:
                st.json(resp.json())
        except httpx.HTTPError as exc:
            st.error(f"API unreachable: {exc}")

st.divider()
if st.button("Health", type="secondary"):
    try:
        st.json(httpx.get(f"{API}/v1/health", timeout=10).json())
    except httpx.HTTPError as exc:
        st.error(f"API unreachable: {exc}")