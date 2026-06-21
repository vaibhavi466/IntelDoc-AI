# pyrefly: ignore [missing-import]
import torch  # Must be imported first to avoid WinError 1114 DLL initialization conflict on Windows
import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image
# pyrefly: ignore [missing-import]
from wordcloud import WordCloud
import pandas as pd
import json

# Import our custom modules
from src.inference import predict_document
from src.extraction import extract_information
from src.summarization import generate_summary
from src.utils import (
    init_db, 
    save_to_db, 
    get_db_history, 
    calculate_text_metrics, 
    delete_db_entries, 
    convert_pdf_to_image
)

# Initialize the database immediately
init_db()

# 1. Page Config
st.set_page_config(page_title="IntelDoc AI", page_icon="📄", layout="wide")

# 2. Custom CSS for "Neon/Dark" Look
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #374151;
        }

        /* Card Styling */
        div[data-testid="stMetric"] {
            background-color: #1F2937;
            border: 1px solid #374151;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetric"] label {
            color: #9CA3AF !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F3F4F6 !important;
        }

        /* Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.3);
        }

        /* File Uploader Styling */
        div[data-testid="stFileUploader"] {
            border: 1px dashed #4B5563;
            border-radius: 10px;
            padding: 20px;
            background-color: #1F2937;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #8B5CF6;
        }

        h1, h2, h3 {
            color: #F3F4F6 !important;
        }
        
        div[data-testid="stTable"] {
            color: #E5E7EB;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# 3. Sidebar Navigation
with st.sidebar:
    st.title("IntelDoc AI")
    st.markdown("---")
    page = st.radio("Navigate to:", ["Analysis Dashboard", "History Log", "System Analytics"])
    st.markdown("---")
    st.caption("v1.0 | Powered by DistilBERT & SpaCy")

# PAGE 1: ANALYSIS DASHBOARD

if page == "Analysis Dashboard":
    st.markdown("""
    <h1 style='text-align: center; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    Intelligent Document Analysis
    </h1>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # --- UPLOAD SECTION ---
    uploaded_file = st.file_uploader(" Upload Document (PDF, JPG, PNG)", type=["jpg", "png", "jpeg", "tif", "pdf"])

    if uploaded_file is not None:
        # Clear previous results when a new file is uploaded
        if st.session_state.get('last_file') != uploaded_file.name:
            for key in ['analyzed', 'label', 'confidence', 'text', 'summary']:
                st.session_state.pop(key, None)
            st.session_state['last_file'] = uploaded_file.name
        st.markdown("---")
        
        # 1. SAVE THE FILE (Required for conversion)
        if not os.path.exists("data"):
            os.makedirs("data")
            
        file_ext = uploaded_file.name.split(".")[-1].lower()
        temp_path = os.path.join("data", f"temp_upload.{file_ext}")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 2. HANDLE PDF vs IMAGE
        display_image_path = temp_path 
        
        if file_ext == "pdf":
            with st.spinner("Converting PDF to Image for AI..."):
                converted_path = convert_pdf_to_image(temp_path)
                if converted_path:
                    display_image_path = converted_path
                else:
                    st.error("Failed to convert PDF.")
                    st.stop()

        # Layout: Image on Left, Actions on Right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(display_image_path)
            st.image(image, caption='Document Preview', use_column_width=True)
            
        with col2:
            st.subheader("Processing Options")
            st.info("Document detected. Ready to analyze.")
            
            # Primary Action Button
            analyze_btn = st.button('Analyze & Archive Document', use_container_width=True, type="primary")

        # Logic
        if analyze_btn:
            with st.spinner('Scanning & Processing...'):
                
                # Predict using the IMAGE path
                label, confidence, extracted_text = predict_document(display_image_path)
                if confidence == 0.0 and extracted_text == "":
                    st.error(label)
                    st.stop()
                
                # Generate Summary
                summary = generate_summary(extracted_text, label)

                # Save to Database
                db_msg = save_to_db(uploaded_file, label, confidence, extracted_text, summary)
                st.toast(db_msg, icon="🗄️")
                
                # Save to Session State
                st.session_state['analyzed'] = True
                st.session_state['label'] = label
                st.session_state['confidence'] = confidence
                st.session_state['text'] = extracted_text
                st.session_state['summary'] = summary 

        # --- RESULTS SECTION ---
        if st.session_state.get('analyzed'):
            st.markdown("---")
            
            # 1. Classification Banner
            st.subheader("1. Classification Result")
            st.markdown(f"""
            <div style="background-color: #3730A3; padding: 15px; border-radius: 10px; border-left: 5px solid #818CF8;">
                <h3 style="margin:0; color: white;">Category: {st.session_state['label'].upper()}</h3>
                <p style="margin:0; color: #C7D2FE;">Confidence Score: {st.session_state['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") 

            # 2. Text Metrics
            st.subheader("2. Text Metrics")
            metrics = calculate_text_metrics(st.session_state['text'])
            if metrics:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Word Count", metrics["Word Count"])
                c2.metric("Sentences", metrics["Sentence Count"])
                c3.metric("Avg Word Len", metrics["Avg Word Length"])
                c4.metric("Readability", metrics["Readability Score (ARI)"])
            
            st.write("") 
            
            # Define Columns Correctly BEFORE using them
            col_left, col_right = st.columns(2)
            
            # 3. Extraction (Left)
            with col_left:
                st.subheader("3. Extracted Entities")
                details = extract_information(st.session_state['text'], st.session_state['label'])
                if details:
                    # st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
                    clean_details = {
                        key: (", ".join(value) if isinstance(value, list) else value)
                        for key, value in details.items()
                    }

                    df = pd.DataFrame(list(clean_details.items()), columns=["Field", "Value"])
                    st.table(df)
                else:
                    st.warning("No specific patterns found.")

            # 4. Summarization (Right)
            with col_right:
                st.subheader("4. AI Summary")
                if 'summary' in st.session_state:
                    st.success(st.session_state['summary'])
                else:
                    st.info("Summary available after analysis.")

            # 5. Visual Analytics
            st.subheader("5. Context Cloud")
            try:
                wordcloud = WordCloud(width=1000, height=400, background_color='#1F2937', colormap='cool').generate(st.session_state['text'])
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_alpha(0)
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except ValueError as e:
                st.info("Not enough text for a word cloud.")


# PAGE 2: HISTORY LOG

elif page == "History Log":
    st.title("Database History")
    st.write("Manage your archived document logs below.")
    
    df_history = get_db_history()
    
    if not df_history.empty:
        df_history.insert(0, "Select", False)
        
        edited_df = st.data_editor(
            df_history,
            column_config={
                "Select": st.column_config.CheckboxColumn("Delete?", width="small"),
                "upload_date": st.column_config.DatetimeColumn("Date", format="D MMM YYYY, h:mm a"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1.0, format="%.2f"),
                "filename": st.column_config.TextColumn("Filename"),
                "category": st.column_config.TextColumn("Category"),
                "summary": st.column_config.TextColumn("Summary", width="medium"),
            },
            disabled=["upload_date", "filename", "category", "confidence", "summary"],
            use_container_width=True,
            hide_index=True,
            height=600
        )
        
        rows_to_delete = edited_df[edited_df.Select].index
        
        if len(rows_to_delete) > 0:
            st.warning(f"You have selected {len(rows_to_delete)} document(s) to delete.")
            if st.button("Delete Selected", type="primary"):
                # Get the actual Database IDs
                ids_to_delete = edited_df.loc[rows_to_delete, "id"].tolist()
                success = delete_db_entries(ids_to_delete)
                if success:
                    st.success("Entries deleted successfully!")
                    st.rerun()
                else:
                    st.error("Error deleting entries.")
    else:
        st.info("No documents processed yet.")


# PAGE 3: ANALYTICS

elif page == "System Analytics":
    st.markdown("""
    <h1 style='text-align: center; background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    System Analytics
    </h1>
    """, unsafe_allow_html=True)

    st.write("")

    # ── Section 1: Live Upload Analytics ────────────────────────────────────────
    st.subheader("Upload Analytics")
    df_history = get_db_history()

    if not df_history.empty:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Uploads by Category**")
            st.bar_chart(df_history["category"].value_counts())
        with colB:
            st.markdown("**AI Confidence Trend**")
            try:
                st.line_chart(df_history["confidence"])
            except ValueError:
                st.info("Insufficient data for trend analysis.")
    else:
        st.info("No data available. Process some documents first!")

    st.markdown("---")

    # ── Section 2: Model Evaluation Artifacts ───────────────────────────────────
    st.subheader("Model Evaluation Artifacts")
    st.caption("Run `python -m src.evaluate` after retraining or changing the test dataset.")
    st.write("")

    eval_path = os.path.join("models", "eval_metrics.json")

    try:
        with open(eval_path) as _f:
            _eval = json.load(_f)

        macro_f1  = _eval.get("macro_f1")
        accuracy  = _eval.get("accuracy")
        conf_mat  = _eval.get("confusion_matrix")   # list[list[int]]
        cls_report = _eval.get("class_report")       # dict: {label: {precision, recall, f1-score, support}}

        # ── 2a. Top-level Metric Cards ───────────────────────────────────────
        m1, m2, m3 = st.columns(3)

        with m1:
            if macro_f1 is not None:
                st.markdown(f"""
                <div style="background-color:#1F2937; border:1px solid #374151; padding:18px;
                            border-radius:10px; text-align:center;">
                    <p style="color:#9CA3AF; margin:0; font-size:13px; font-weight:600;
                               letter-spacing:0.05em;">MACRO F1 SCORE</p>
                    <p style="color:#A78BFA; margin:6px 0 0; font-size:32px;
                               font-weight:700;">{macro_f1:.4f}</p>
                    <p style="color:#6B7280; margin:4px 0 0; font-size:11px;">
                        Macro-averaged across all classes</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("`macro_f1` not found in eval_metrics.json.")

        with m2:
            if accuracy is not None:
                colour = "#34D399" if accuracy >= 0.85 else ("#FBBF24" if accuracy >= 0.70 else "#F87171")
                st.markdown(f"""
                <div style="background-color:#1F2937; border:1px solid #374151; padding:18px;
                            border-radius:10px; text-align:center;">
                    <p style="color:#9CA3AF; margin:0; font-size:13px; font-weight:600;
                               letter-spacing:0.05em;">ACCURACY</p>
                    <p style="color:{colour}; margin:6px 0 0; font-size:32px;
                               font-weight:700;">{accuracy:.2%}</p>
                    <p style="color:#6B7280; margin:4px 0 0; font-size:11px;">
                        Overall classification accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("`accuracy` not found in eval_metrics.json.")

        with m3:
            if cls_report:
                total_support = sum(
                    v.get("support", 0)
                    for k, v in cls_report.items()
                    if k not in ("accuracy", "macro avg", "weighted avg")
                )
                num_classes = len([
                    k for k in cls_report
                    if k not in ("accuracy", "macro avg", "weighted avg")
                ])
                st.markdown(f"""
                <div style="background-color:#1F2937; border:1px solid #374151; padding:18px;
                            border-radius:10px; text-align:center;">
                    <p style="color:#9CA3AF; margin:0; font-size:13px; font-weight:600;
                               letter-spacing:0.05em;">EVALUATION DATASET</p>
                    <p style="color:#60A5FA; margin:6px 0 0; font-size:32px;
                               font-weight:700;">{total_support:,}</p>
                    <p style="color:#6B7280; margin:4px 0 0; font-size:11px;">
                        Samples across {num_classes} classes</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No class report data available.")

        st.write("")
        st.markdown("---")

        # ── 2b. Confusion Matrix ─────────────────────────────────────────────
        if conf_mat and cls_report:
            st.subheader("Confusion Matrix")

            class_labels = [
                k for k in cls_report
                if k not in ("accuracy", "macro avg", "weighted avg")
            ]

            import numpy as np
            import matplotlib.ticker as ticker

            cm_array = np.array(conf_mat)

            fig_cm, ax_cm = plt.subplots(
                figsize=(max(6, len(class_labels) * 1.1), max(5, len(class_labels) * 0.9))
            )
            fig_cm.patch.set_facecolor("#111827")
            ax_cm.set_facecolor("#1F2937")

            # Normalise for colour intensity; keep raw counts as labels
            cm_norm = cm_array.astype(float) / (cm_array.sum(axis=1, keepdims=True) + 1e-9)

            im = ax_cm.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1, aspect="auto")

            # Cell annotations
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    val = cm_array[i, j]
                    text_colour = "white" if cm_norm[i, j] > 0.55 else "#C4B5FD"
                    ax_cm.text(j, i, str(val), ha="center", va="center",
                               fontsize=9, color=text_colour, fontweight="bold")

            # Axes
            ax_cm.set_xticks(range(len(class_labels)))
            ax_cm.set_yticks(range(len(class_labels)))
            ax_cm.set_xticklabels(class_labels, rotation=35, ha="right",
                                   fontsize=8, color="#D1D5DB")
            ax_cm.set_yticklabels(class_labels, fontsize=8, color="#D1D5DB")
            ax_cm.set_xlabel("Predicted Label", color="#9CA3AF", fontsize=10, labelpad=10)
            ax_cm.set_ylabel("True Label", color="#9CA3AF", fontsize=10, labelpad=10)
            ax_cm.tick_params(colors="#4B5563")
            for spine in ax_cm.spines.values():
                spine.set_edgecolor("#374151")

            cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.03, pad=0.04)
            cbar.ax.tick_params(colors="#9CA3AF", labelsize=8)
            cbar.set_label("Normalised", color="#9CA3AF", fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        elif not conf_mat:
            st.info("`confusion_matrix` not found in eval_metrics.json.")

        st.markdown("---")

        # ── 2c. Class-wise Performance ───────────────────────────────────────
        if cls_report:
            st.subheader("Class-wise Performance")

            # Build a clean DataFrame (exclude aggregate rows)
            excluded_keys = {"accuracy", "macro avg", "weighted avg"}
            rows = []
            for label, metrics_dict in cls_report.items():
                if label in excluded_keys:
                    continue
                rows.append({
                    "Class": label,
                    "Precision": round(metrics_dict.get("precision", 0), 4),
                    "Recall":    round(metrics_dict.get("recall", 0), 4),
                    "F1 Score":  round(metrics_dict.get("f1-score", 0), 4),
                    "Support":   int(metrics_dict.get("support", 0)),
                })
            df_cls = pd.DataFrame(rows).set_index("Class")

            # Styled table
            def _colour_cell(val):
                if isinstance(val, float):
                    if val >= 0.85:
                        bg, fg = "#064E3B", "#6EE7B7"
                    elif val >= 0.70:
                        bg, fg = "#78350F", "#FCD34D"
                    else:
                        bg, fg = "#7F1D1D", "#FCA5A5"
                    return f"background-color:{bg}; color:{fg}; font-weight:600;"
                return ""

            styled_df = (
                df_cls.style
                .map(_colour_cell, subset=["Precision", "Recall", "F1 Score"])
                .format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"})
                .set_table_styles([
                    {"selector": "th",
                     "props": [("background-color", "#1F2937"),
                               ("color", "#9CA3AF"),
                               ("font-size", "12px"),
                               ("padding", "8px 12px"),
                               ("border-bottom", "1px solid #374151")]},
                    {"selector": "td",
                     "props": [("background-color", "#111827"),
                               ("color", "#E5E7EB"),
                               ("padding", "8px 12px"),
                               ("border-bottom", "1px solid #1F2937")]},
                    {"selector": "tr:hover td",
                     "props": [("background-color", "#1F2937")]},
                ])
            )
            st.dataframe(styled_df, use_container_width=True)

            st.write("")

            # Grouped bar chart — Precision / Recall / F1
            st.markdown("**Per-class metric comparison**")

            fig_bar, ax_bar = plt.subplots(figsize=(max(8, len(df_cls) * 1.4), 4.5))
            fig_bar.patch.set_facecolor("#111827")
            ax_bar.set_facecolor("#1F2937")

            x       = np.arange(len(df_cls))
            width   = 0.26
            labels  = df_cls.index.tolist()
            colours = {"Precision": "#818CF8", "Recall": "#34D399", "F1 Score": "#FBBF24"}

            for i, (metric, colour) in enumerate(colours.items()):
                offset = (i - 1) * width
                bars = ax_bar.bar(x + offset, df_cls[metric], width,
                                  label=metric, color=colour, alpha=0.88,
                                  edgecolor="#111827", linewidth=0.6)
                for bar in bars:
                    h = bar.get_height()
                    if h > 0.05:
                        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                                    f"{h:.2f}", ha="center", va="bottom",
                                    fontsize=7, color="#D1D5DB")

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(labels, rotation=30, ha="right",
                                    fontsize=9, color="#D1D5DB")
            ax_bar.set_ylim(0, 1.15)
            ax_bar.set_ylabel("Score", color="#9CA3AF", fontsize=10)
            ax_bar.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax_bar.tick_params(axis="y", colors="#6B7280", labelsize=8)
            ax_bar.tick_params(axis="x", colors="#6B7280")
            for spine in ax_bar.spines.values():
                spine.set_edgecolor("#374151")
            ax_bar.axhline(0.85, color="#374151", linewidth=0.8,
                           linestyle="--", alpha=0.7)
            ax_bar.text(len(x) - 0.4, 0.86, "target (0.85)",
                        fontsize=7, color="#6B7280")

            leg = ax_bar.legend(frameon=True, facecolor="#1F2937",
                                edgecolor="#374151", labelcolor="#D1D5DB",
                                fontsize=9, loc="upper right")

            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        elif not cls_report:
            st.info("`class_report` not found in eval_metrics.json.")

    except FileNotFoundError:
        st.warning("No evaluation metrics found. Run `python src/evaluate.py` to generate them.")
    except json.JSONDecodeError:
        st.error("`models/eval_metrics.json` is malformed. Please regenerate it.")