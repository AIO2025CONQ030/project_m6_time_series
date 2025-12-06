from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import time
from utils import ModelInfo, line_chart, plot_forecast_with_confidence
import seaborn as sns

st.set_page_config(layout="wide")

st.session_state["df"] = None
st.session_state["training_progress"] = 0
st.session_state["model_trained"] = False
st.session_state["models"] = ["dlinear", "nlinear", "hybrid", "gru", "lstm", "arima"]
st.session_state["redraw"] = True
st.session_state["redraw_forecast"] = True

st.markdown(
    "<h2 style='text-align:center;'>Project - CONQ030</h2>", unsafe_allow_html=True
)

left, right = st.columns([0.4, 0.6], border=True)
with left:
    st.header("Data Preview")
    df = pd.read_csv("FPT_train.csv")
    st.dataframe(df, use_container_width=True)
    st.session_state["df"] = df
    st.header("Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.header("Coerrelation Matrix")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(5, 2)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

    # output_col_param = st.selectbox("Choose output column", key='choose_output_col', options=df.columns.tolist() if df is not None else [])
with right:
    seq_labels = [
        "7Days",
        "15Days",
        "30Days",
        "60Days",
        "120Days",
        "150Days",
        "190Days",
        "240Days",
        "480Days",
    ]
    select_box = st.selectbox(
        "Past Sequence length",
        options=[7, 15, 30, 60],
        format_func=lambda x: seq_labels[[7, 15, 30, 60].index(x)],
        key="seq_len",
        on_change=lambda: st.session_state.update({"redraw": True}),
    )
    df = st.session_state["df"]
    if st.session_state["redraw"]:
        st.header("Accuracy")
        print("Redrawing charts...", st.session_state["seq_len"])
        # line_chart()

        st.header("Train loss vs Validation loss")
        models = st.session_state["models"]
        model_infos = []
        for model in models:
            train_mse = np.loadtxt(
                f"{model}/train_loss{st.session_state['seq_len']}d.csv", delimiter=","
            )
            val_mse = np.loadtxt(
                f"{model}/val_loss{st.session_state['seq_len']}d.csv", delimiter=","
            )
            final_predictions = np.loadtxt(
                f"{model}/submission{st.session_state['seq_len']}d.csv", delimiter=","
            )
            model_info = ModelInfo(
                name=model,
                train_mse=train_mse[0:100],
                val_mse=val_mse[0:100],
                predictions=final_predictions,
            )  # Placeholder for val_mse
            model_infos.append(model_info)

        st.subheader("Train Loss")
        line_chart(
            [y_val.train_mse for y_val in model_infos],
            [y_val.name for y_val in model_infos],
        )

        st.subheader("Validation Loss")
        line_chart(
            [y_val.val_mse for y_val in model_infos],
            [y_val.name for y_val in model_infos],
        )

        st.session_state["redraw"] = False

        st.header("Forecast Predictions")
        st.selectbox(
            "Choose Model for Forecast Plot",
            options=[model.name for model in model_infos],
            key="forecast_model",
        )
        if st.session_state["redraw_forecast"]:
            plot_forecast_with_confidence(
                df=df,
                predictions=model_infos[
                    [model.name for model in model_infos].index(
                        st.session_state["forecast_model"]
                    )
                ].predictions,
                all_predictions=None,
                history_days=150,
                confidence_level=0.90,
                figsize=(16, 8),
            )
        st.session_state["redraw_forecast"] = False

    st.header("Model Evaluation")

    df_eval = pd.DataFrame(
        {
            "STT": range(1, len(model_infos) + 1),
            "Model": [model_info.name for model_info in model_infos],
        }
    )
    df_eval["Train MSE"] = [
        float(np.mean(model_info.train_mse)) for model_info in model_infos
    ]
    df_eval["Validation MSE"] = [
        float(np.mean(model_info.val_mse)) for model_info in model_infos
    ]
    df_eval.reset_index(drop=True, inplace=True)
    
    st.dataframe(df_eval, use_container_width=True, hide_index=True)
