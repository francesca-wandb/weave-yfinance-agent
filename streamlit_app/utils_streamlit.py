import os
import re

import streamlit as st


def render_markdown_with_images(markdown_text: str, image_base_path: str = "."):
    """
    Renders Markdown in Streamlit, extracting and displaying any embedded images.

    Parameters:
    - markdown_text (str): The Markdown string possibly containing image references.
    - image_base_path (str): Base path for relative image links (default is current dir).

    Returns:
    - None (renders content via Streamlit)
    """
    image_pattern = r"!\[.*?\]\((.*?)\)"
    image_paths = re.findall(image_pattern, markdown_text)

    # Remove image markdown tags so st.markdown doesn't try to render broken links
    cleaned_md = re.sub(image_pattern, "", markdown_text)

    # First display cleaned markdown
    st.markdown(cleaned_md, unsafe_allow_html=True)

    # Then display each image
    for img_path in image_paths:
        full_path = os.path.join(image_base_path, img_path)
        if os.path.isfile(full_path):
            st.image(full_path)
        else:
            st.warning(f"Image not found: {img_path}")


def display_log_and_weave_button(weave_url: str):
    """Render a subheader and a button to open the provided Weave URL in a new tab."""

    st.subheader("Weave")
    weave_open_button(weave_url)


def weave_open_button(weave_url: str):
    """
    Render a clickable button in Streamlit to open the provided Weave URL in a new tab.
    """
    st.markdown(
        f"""
        <a href="{weave_url}" target="_blank">
            <button style='font-size:16px;padding:8px 16px;border-radius:8px;'>🍩 Open Weave Trace</button>
        </a>
        """,
        unsafe_allow_html=True,
    )
