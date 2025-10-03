# app.py
import os
import random
import html
import base64
import json
from json import load

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from treefarms import TREEFARMS
import timbertrek


def make_html(decision_paths, width):
    # HTML template for TimberTrek widget
    html_top = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TimberTrek</title>
    <style>
    html{font-size:16px;-moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
    -webkit-text-size-adjust:100%;-moz-text-size-adjust:100%}
    html,body{position:relative;width:100%;height:100%}
    body{margin:0;padding:0;box-sizing:border-box;
    font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Oxygen-Sans,
    Ubuntu,Cantarell,Helvetica Neue,sans-serif;color:#4a4a4a;font-size:1em;
    font-weight:400;line-height:1.5}
    *,:after,:before{box-sizing:inherit}
    a{color:#0064c8;text-decoration:none}
    a:hover{text-decoration:underline}
    a:visited{color:#0050a0}
    label{display:block}
    input,button,select,textarea{font-family:inherit;font-size:inherit;
    -webkit-padding:.4em 0;padding:.4em;margin:0 0 .5em;box-sizing:border-box;
    border:1px solid #ccc;border-radius:2px}
    input:disabled{color:#ccc}
    button{color:#333;background-color:#f4f4f4;outline:none}
    button:disabled{color:#999}
    button:not(:disabled):active{background-color:#ddd}
    button:focus{border-color:#666}
    </style>"""
    html_bottom = """</head><body></body></html>"""

    # Load timbertrek.js from installed package
    js_path = os.path.join(os.path.dirname(timbertrek.__file__), "timbertrek.js")
    with open(js_path, "rb") as f:
        js_b = f.read()

    # Encode the JS & CSS with base64
    js_base64 = base64.b64encode(js_b).decode("utf-8")
    data_json = json.dumps(decision_paths)
    messenger_js = f"""
        (function() {{
            const event = new Event('timbertrekData');
            event.data = {data_json};
            event.width = {width};
            document.dispatchEvent(event);
        }}())
    """
    messenger_js_base64 = base64.b64encode(messenger_js.encode()).decode("utf-8")
    html_str = (
        html_top
        + f"<script defer src='data:text/javascript;base64,{js_base64}'></script>"
        + f"<script defer src='data:text/javascript;base64,{messenger_js_base64}'></script>"
        + html_bottom
    )

    return html.escape(html_str)

def visualize_return_html(decision_paths, width=500, height=650):
    assert isinstance(decision_paths, dict), "`decision_paths` has to be a dictionary."
    assert "trie" in decision_paths, "`decision_paths` is not valid (no `trie` key)."
    assert "featureMap" in decision_paths, "`decision_paths` is not valid (no `featureMap` key)."
    assert "treeMap" in decision_paths, "`decision_paths` is not valid (no `treeMap` key)."

    html_str = make_html(decision_paths, width)

    iframe_id = "timbertrek-iframe-" + str(int(random.random() * 1e8))
    iframe = f"""
        <iframe
            srcdoc="{html_str}"
            frameBorder="0"
            width="100%"
            height="{height}px"
            id="{iframe_id}">
        </iframe>
    """
    return iframe


# --- Streamlit UI ---
st.set_page_config(page_title="TimberTrek Explorer", layout="wide")
st.title("TimberTrek Explorer")

# Sidebar controls
st.sidebar.header("⚙️ Model Parameters")

regularization = st.sidebar.slider(
    "Regularization", min_value=0.0, max_value=1.0, value=0.015, step=0.005, format="%.3f"
)
rashomon_bound_multiplier = st.sidebar.slider(
    "Rashomon Bound Multiplier", min_value=0.0, max_value=1.0, value=0.15, step=0.01
)
depth_budget = st.sidebar.slider(
    "Depth Budget", min_value=1, max_value=10, value=5, step=1
)

train_button = st.sidebar.button("Configure parameters")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded with shape {df.shape}")

    # Prepare X and y
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Config for TREEFARMS
    config = {
        "regularization": regularization,
        "rashomon": True,
        "rashomon_bound_multiplier": rashomon_bound_multiplier,
        "depth_budget": depth_budget,
        "rashomon_trie": "rashomon_trie.json",
        "verbose": True
    }

    # Fit Treefarms model
    st.write("Training TREEFARMS model...")
    model = TREEFARMS(config)
    model.fit(X, y)

    # just print for verification
    rashomon = model.get_tree_count()
    st.success(f"Rashomon set generated with {rashomon} trees")

    feature_names = X.columns.tolist()
    trie = load(open("rashomon_trie.json", "r"))
    decision_paths = timbertrek.transform_trie_to_rules(trie, df, feature_names=feature_names)

    # --- TimberTrek visualization ---
    st.write("### TimberTrek Visualization")
    iframe_html = visualize_return_html(decision_paths, width=550, height=650)
    components.html(iframe_html, height=800, scrolling=True)
