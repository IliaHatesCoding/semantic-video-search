import streamlit as st
from sentence_transformers import SentenceTransformer

# Import backend functions from inference.py (your existing code)
from inference import (
    get_db_connection,
    encode_query,
    search_similar_segments,
    group_results_by_video,
    format_duration,
    format_number,
    MODEL_NAME,
)

# ------------- CACHED MODEL LOADER ------------- #
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


# ------------- CATEGORY CONFIGURATION ------------- #
CATEGORY_OPTIONS = {
    "Speeches of politicians": ["Any", "Donald Trump", "Vladimir Putin", "Xi Jinping"],
    "Movies": ["Any"],
    "News clips": ["Any"],
    "Music clips": ["Any"],
    "Sport clips": ["Any"],
}


# ------------- MAIN APP ------------- #
def main():
    st.set_page_config(
        page_title="Semantic Video Search",
        page_icon="🎬",
        layout="wide",
    )

    # ---- Simple custom styling ----
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #555555;
            margin-bottom: 2rem;
        }
        .video-card {
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid #e5e5e5;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            margin-bottom: 2rem;
        }
        .similarity-badge {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background-color: #e6f4ff;
            color: #0056b3;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main page: Title + tagline
    st.markdown(
        '<div class="main-title">🧠 Speech Explorer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Semantic search in political speeches, news and more – type a phrase and jump straight to the moment it appears.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: categories
    st.sidebar.title("Categories")

    main_category = st.sidebar.selectbox(
        "Main category",
        list(CATEGORY_OPTIONS.keys()),
    )

    subcategories = CATEGORY_OPTIONS.get(main_category, ["Any"])
    sub_category = st.sidebar.selectbox("Sub category", subcategories)

    st.sidebar.markdown("---")
    st.sidebar.caption("You can adjust categories/subcategories later.")

    # Search controls
    query = st.text_input("Search phrase or word (e.g. 'China')", "")

    col_left, col_right = st.columns(2)
    with col_left:
        min_similarity = st.slider(
            "Minimum similarity (match quality)",
            min_value=0.3,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Higher = only very close matches, lower = more results but weaker matches.",
        )
    with col_right:
        max_results = st.slider(
            "Maximum number of segments",
            min_value=20,
            max_value=300,
            value=100,
            step=20,
        )

    search_button = st.button("🔍 Search")

    if search_button:
        if not query.strip():
            st.warning("Please enter a search phrase first.")
            return

        # Show info about current search
        st.markdown(
            f"**Searching for:** ` {query} `  \n"
            f"**Category:** {main_category} → {sub_category}"
        )
        st.markdown("---")

        with st.spinner("Loading model and searching in the database..."):
            # 1) Load model (cached)
            model = load_model()

            # 2) DB connection
            conn = get_db_connection()

            try:
                # 3) Encode query
                query_embedding = encode_query(model, query)

                # 4) pgvector search
                results = search_similar_segments(
                    conn,
                    query_embedding,
                    num_results=max_results,
                    min_similarity=min_similarity,
                )

                # 5) Optional: filter by politician
                if (
                    main_category == "Speeches of politicians"
                    and sub_category != "Any"
                ):
                    name = sub_category.lower()
                    filtered = []
                    for r in results:
                        title = (r.get("title") or "").lower()
                        desc = (r.get("description") or "").lower()
                        if name in title or name in desc:
                            filtered.append(r)
                    results = filtered

                if not results:
                    st.info(
                        "No results found with the current settings "
                        f"(min similarity {min_similarity:.2f}). Try lowering the threshold or changing the query."
                    )
                    return

                # 6) Group by video
                video_groups = group_results_by_video(results)

            finally:
                conn.close()

        # Display results
        total_segments = len(results)
        unique_videos = len(video_groups)

        st.success(
            f"Found {unique_videos} videos with {total_segments} matching segments "
            f"(similarity ≥ {min_similarity*100:.0f}%)."
        )
        st.write("Scroll down to see the clips 👇")
        st.markdown("---")

        for video_idx, (video_id, video_data) in enumerate(
            video_groups.items(), start=1
        ):
            video_info = video_data["video_info"]
            segments = video_data["segments"]
            best_segment = segments[0]

            title = video_info.get("title") or "Untitled video"
            description = video_info.get("description") or ""
            views = format_number(video_info.get("view_count"))
            likes = format_number(video_info.get("like_count"))
            language = video_info.get("transcription_language") or "Unknown"
            published = video_info.get("published_at")

            start_sec = int(best_segment.get("segment_start") or 0)
            end_sec = int(best_segment.get("segment_end") or 0)
            duration_txt = f"{format_duration(start_sec)} - {format_duration(end_sec)}"

            similarity_pct = best_segment.get("similarity", 0) * 100

            watch_url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"

            # Card layout
            with st.container():
                st.markdown('<div class="video-card">', unsafe_allow_html=True)
                st.subheader(f"{video_idx}. {title}")

                vcol, icol = st.columns([2, 3])

                with vcol:
                    st.video(watch_url)

                with icol:
                    st.markdown("**Best matching segment:**")
                    st.write(best_segment.get("segment_text") or "(no text)")

                    st.markdown(
                        f'<span class="similarity-badge">🎯 {similarity_pct:.1f}% match</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"- ⏱ **Timeframe:** {duration_txt}  \n"
                        f"- 🌍 **Language:** {language}"
                    )
                    st.markdown(
                        f"- 👁 **Views:** {views}  \n"
                        f"- 👍 **Likes:** {likes}"
                    )
                    if published:
                        st.markdown(f"- 📅 **Published:** {published}")

                    st.markdown(
                        f"[▶️ Watch on YouTube]({watch_url})",
                        unsafe_allow_html=False,
                    )

                # Additional segments
                if len(segments) > 1:
                    with st.expander(
                        f"Show {len(segments) - 1} more matching segment(s) from this video"
                    ):
                        for seg in segments[1:]:
                            seg_start = int(seg.get("segment_start") or 0)
                            seg_end = int(seg.get("segment_end") or 0)
                            seg_similarity_pct = seg.get("similarity", 0) * 100
                            seg_url = (
                                f"https://www.youtube.com/watch?v={video_id}"
                                f"&t={seg_start}s"
                            )

                            st.markdown(
                                f"**Similarity:** {seg_similarity_pct:.1f}%  \n"
                                f"**Time:** {format_duration(seg_start)} - {format_duration(seg_end)}"
                            )
                            st.write(seg.get("segment_text") or "(no text)")
                            st.markdown(f"[Open this moment on YouTube]({seg_url})")
                            st.markdown("---")

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("---")


if __name__ == "__main__":
    main()    
