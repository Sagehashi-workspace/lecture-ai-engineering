# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL1_NAME, MODEL2_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_models():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # モデル1
        pipe1 = pipeline("text-generation", model=MODEL1_NAME, model_kwargs={"torch_dtype": torch.bfloat16}, device=device)
        st.success(f"モデル '{MODEL1_NAME}' の読み込みに成功しました。")
        
        # モデル2
        pipe2 = pipeline("text-generation", model=MODEL2_NAME, model_kwargs={"torch_dtype": torch.bfloat16}, device=device)
        st.success(f"モデル '{MODEL2_NAME}' の読み込みに成功しました。")
        
        return pipe1, pipe2
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None, None
# 両方のモデルをロード
pipe1, pipe2 = llm.load_models()

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.write("左のサイドバーから使用するモデルを選択してください。使用できるモデルは以下の2つです。")
st.code("google/gemma-2-2b-jpn-it")
st.write("デフォルトで選択。日本語テキスト向けに微調整されたGemma 2 2Bモデルです。Gemma 2における英語のみのクエリと同等のパフォーマンスで日本語をサポートします。")
st.code("google/gemma-3-12b-it")
st.write("gemmaの最新モデルです。128K の大規模なコンテキストウィンドウと、140 を超える言語での多言語サポートを備え、以前のバージョンよりも多くのサイズで利用できます。")
st.markdown("---")

# --- モデル選択 ---
model_options = [
    "google/gemma-2-2b-jpn-it",  # 既存モデル
    "google/gemma-3-12b-it"      # 新しいモデル
]

selected_model = st.sidebar.selectbox("使用するモデルを選択", model_options)

# --- 使用するモデルを選択 ---
if selected_model == "google/gemma-2-2b-jpn-it":
    pipe = pipe1
elif selected_model == "google/gemma-3-12b-it":
    pipe = pipe2

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: Sagehashi")