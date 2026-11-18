"""Free chat page with local model backend integration."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
import streamlit as st

from utils.api_client import PhilTraceClient

BACKEND_URL = "http://localhost:8000"


def init_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_client" not in st.session_state:
        st.session_state.chat_client = PhilTraceClient()
    if "philosophy_mode" not in st.session_state:
        st.session_state.philosophy_mode = True
    if "chat_provider" not in st.session_state:
        st.session_state.chat_provider = "ollama"


def render_header() -> None:
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
          <h1 class="phil-title">Philosophy Chat</h1>
          <p class="phil-subtitle">在本地模型环境中进行可控、可追踪的哲学对话。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _fetch_model_status(client: PhilTraceClient) -> Tuple[bool, List[str], str]:
    try:
        payload = client._request("GET", "/models").json()
    except Exception as exc:  # noqa: BLE001
        return False, [], str(exc)

    available = bool(payload.get("available", False))
    models = payload.get("models", []) if available else []

    model_names: List[str] = []
    for model in models:
        if isinstance(model, dict):
            model_names.append(model.get("name", model.get("id", "unknown")))
        else:
            model_names.append(str(model))

    return available, model_names, payload.get("error", "")


def render_settings_panel() -> Tuple[str, Optional[str], bool, bool]:
    st.markdown('<div class="settings-panel"><div class="settings-title">对话设置</div>', unsafe_allow_html=True)

    # Provider selection
    provider = st.radio(
        "LLM 服务",
        options=["ollama", "deepseek"],
        format_func=lambda x: {"ollama": "本地 Ollama", "deepseek": "DeepSeek API"}[x],
        horizontal=True,
        key="chat_provider_radio",
    )
    st.session_state.chat_provider = provider

    # DeepSeek API key input
    if provider == "deepseek":
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            key="chat_deepseek_key",
            help="输入 DeepSeek API Key（也可通过环境变量 DEEPSEEK_API_KEY 设置）",
        )
        st.session_state.deepseek_api_key = deepseek_key
        model = st.selectbox(
            "DeepSeek 模型",
            options=["deepseek-chat", "deepseek-reasoner"],
            index=0,
            key="deepseek_model_selector",
            help="选择 DeepSeek 模型",
        )
        available = True  # DeepSeek is always "available" if key is set
    else:
        # Ollama model selection
        client = st.session_state.chat_client
        available, model_names, error_message = _fetch_model_status(client)

        if not model_names:
            model_names = ["qwen2:7b"]

        if not available:
            st.warning("未检测到可用 Ollama 服务。请先运行 `ollama serve`。")
            if error_message:
                st.caption(f"详细信息: {error_message}")

        model = st.selectbox(
            "选择模型",
            options=model_names,
            index=0,
            disabled=not available,
            help="从后端可用模型中选择当前对话模型。",
            key="model_selector",
        )

    lora_options = {
        "无": None,
        "Spinoza": "spinoza",
        "Kant": "kant",
        "Hegel": "hegel",
    }
    selected_lora_name = st.selectbox(
        "哲学风格",
        options=list(lora_options.keys()),
        index=0,
        key="lora_selector",
        help="用于切换系统提示词风格。",
    )

    philosophy_mode = st.toggle(
        "启用哲学深度模式",
        value=st.session_state.philosophy_mode,
        key="philosophy_toggle",
        help="关闭后将不加载风格提示词。",
    )

    st.session_state.philosophy_mode = philosophy_mode
    st.session_state.current_model = model
    st.session_state.current_lora = lora_options[selected_lora_name]

    st.markdown("</div>", unsafe_allow_html=True)
    return model, lora_options[selected_lora_name], philosophy_mode, available


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="welcome-message">
          <div class="welcome-icon">●</div>
          <div class="welcome-title">开始对话</div>
          <div class="welcome-hint">先在上方选择模型与风格，然后输入你的哲学问题。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def call_chat_api_stream(
    base_url: str,
    message: str,
    model: str,
    adapter: Optional[str],
    history: Optional[List[Dict[str, str]]],
):
    """Stream chat response from Ollama backend."""
    payload = {
        "message": message,
        "model": model,
        "adapter": adapter,
        "history": history,
        "stream": True,
    }

    with httpx.Client(timeout=600.0) as client:
        with client.stream("POST", f"{base_url}/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("done"):
                        break
                    if chunk := data.get("chunk"):
                        yield chunk


def call_deepseek_stream(
    message: str,
    api_key: str,
    model: str = "deepseek-chat",
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
):
    """Stream chat response from DeepSeek via backend proxy."""
    payload: Dict[str, Any] = {
        "message": message,
        "api_key": api_key,
        "model": model,
    }
    if history:
        payload["history"] = history
    if system_prompt:
        payload["system_prompt"] = system_prompt

    with httpx.Client(timeout=600.0) as client:
        with client.stream("POST", f"{BACKEND_URL}/chat/deepseek", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("done"):
                        if error := data.get("error"):
                            raise RuntimeError(error)
                        break
                    if chunk := data.get("chunk"):
                        yield chunk


def render_chat_view() -> None:
    init_chat_state()
    render_header()

    selected_model, selected_lora, philosophy_mode, backend_ready = render_settings_panel()

    if not st.session_state.chat_messages:
        render_empty_state()
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))

    prompt = st.chat_input("请输入你的问题...", disabled=not backend_ready, key="chat_input")
    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    user_msg = {"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}
    st.session_state.chat_messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.chat_messages[:-1]
        if item.get("role") in {"user", "assistant"}
    ]

    provider = st.session_state.get("chat_provider", "ollama")

    with st.chat_message("assistant"):
        try:
            message_placeholder = st.empty()
            full_response = ""

            if provider == "deepseek":
                api_key = st.session_state.get("deepseek_api_key", "")
                if not api_key:
                    import os
                    api_key = os.getenv("DEEPSEEK_API_KEY", "")
                if not api_key:
                    raise ValueError("请输入 DeepSeek API Key 或设置环境变量 DEEPSEEK_API_KEY")

                for chunk in call_deepseek_stream(
                    message=prompt,
                    api_key=api_key,
                    model=selected_model,
                    history=history if history else None,
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            else:
                client = st.session_state.chat_client
                for chunk in call_chat_api_stream(
                    base_url=client.base_url,
                    message=prompt,
                    model=selected_model,
                    adapter=selected_lora if philosophy_mode else None,
                    history=history if history else None,
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": full_response, "timestamp": datetime.now().isoformat()}
            )
        except httpx.HTTPStatusError as exc:
            text = f"请求失败 ({exc.response.status_code})：{exc.response.text}"
            st.error(text)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": text, "timestamp": datetime.now().isoformat()}
            )
        except httpx.ConnectError:
            text = "无法连接后端服务。请启动后端：uvicorn backend.main:app --reload"
            st.error(text)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": text, "timestamp": datetime.now().isoformat()}
            )
        except Exception as exc:  # noqa: BLE001
            text = f"发生异常：{exc}"
            st.error(text)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": text, "timestamp": datetime.now().isoformat()}
            )

    st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="PhilTrace - Free Chat", layout="wide")
    from ui.style import apply_all_styles

    apply_all_styles()
    render_chat_view()
