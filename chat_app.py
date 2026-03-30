"""
大模型多轮对话界面
使用 Streamlit 实现，支持历史对话保存和加载
"""

import os
import json
import openai
import streamlit as st
from datetime import datetime
from pathlib import Path

# 历史对话保存目录
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)


def get_history_files():
    """获取所有历史对话文件列表"""
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return files


def save_conversation(messages, current_filename=None, last_user_prompt="", force_new=False):
    """保存对话到文件
    
    Args:
        messages: 对话消息列表
        current_filename: 当前文件名（如果是新建对话则为None）
        last_user_prompt: 用户最后的问题，用于生成文件名
        force_new: 是否强制创建新文件（另存为）
    """
    # 只有新建对话（没有当前文件）或强制新建时，才生成新文件名
    if current_filename is None or force_new:
        # 使用用户最后的问题作为文件名，清理非法字符
        if last_user_prompt:
            # 取前30个字符作为文件名，去除非法字符
            safe_name = "".join(c for c in last_user_prompt if c.isalnum() or c in "_- ").strip()
            safe_name = safe_name[:30] if len(safe_name) > 30 else safe_name
            if not safe_name:
                safe_name = "新对话"
        else:
            safe_name = "新对话"
        
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = f"{safe_name}_{timestamp}.json"
    else:
        # 使用当前文件名（覆盖保存）
        filename = current_filename
    
    filepath = HISTORY_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    return filename


def load_conversation(filename):
    """从文件加载对话"""
    filepath = HISTORY_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def init_client():
    """初始化 OpenAI 客户端"""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        return None
    try:
        client = openai.Client(
            base_url="https://api.moonshot.cn/v1",
            api_key=api_key,
            timeout=60.0
        )
        return client
    except Exception as e:
        st.error(f"初始化客户端失败: {e}")
        return None


def stream_chat_completion(client, messages, model="kimi-k2.5", temperature=1.0, max_tokens=16384):
    """流式调用大模型"""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        
        thinking = False
        reasoning_content = ""
        content = ""
        
        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta and hasattr(choice.delta, "reasoning_content"):
                    if not thinking:
                        thinking = True
                    reasoning_piece = getattr(choice.delta, "reasoning_content")
                    reasoning_content += reasoning_piece
                    yield {"type": "reasoning", "content": reasoning_piece}
                if choice.delta and choice.delta.content:
                    if thinking:
                        thinking = False
                    content_piece = choice.delta.content
                    content += content_piece
                    yield {"type": "content", "content": content_piece}
    except Exception as e:
        # 重新抛出异常，带上更多上下文
        raise Exception(f"API调用失败: {str(e)}") from e


def main():
    st.set_page_config(page_title="私人用大模型聊天机器人", page_icon="🤖", layout="wide")
    
    # 初始化 session state（必须在侧边栏之前）
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    st.title("🤖 私人用大模型聊天机器人")
    st.caption("👈 点击左上角箭头展开侧边栏，查看历史对话和设置")
    
    # 侧边栏：历史对话管理
    with st.sidebar:
        st.header("历史对话")
        
        # 新建对话按钮
        if st.button("➕ 新建对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_file = None
            st.rerun()
        
        st.divider()
        
        # 加载历史对话
        history_files = get_history_files()
        if history_files:
            st.subheader("选择历史对话")
            for file in history_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    display_name = file.stem
                    if st.button(f"📄 {display_name}", key=f"load_{file.name}", use_container_width=True):
                        st.session_state.messages = load_conversation(file.name)
                        st.session_state.current_file = file.name
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{file.name}"):
                        file.unlink()
                        if st.session_state.get("current_file") == file.name:
                            st.session_state.messages = []
                            st.session_state.current_file = None
                        st.rerun()
        else:
            st.info("暂无历史对话")
        
        # API Key 检查
        st.divider()
        st.header("API 状态")
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if api_key:
            st.success("✅ API Key 已设置")
        else:
            st.error("❌ API Key 未设置")
        
        # 系统提示词设置
        st.divider()
        st.header("系统提示词")
        
        # 默认系统提示词
        default_system_prompt = "你是 Kimi，一个有帮助的 AI 助手。请用中文回答用户的问题。"
        
        # 初始化系统提示词
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = default_system_prompt
        
        # 系统提示词输入框
        system_prompt = st.text_area(
            "设置 AI 的行为和角色",
            value=st.session_state.system_prompt,
            height=100,
            help="系统提示词会影响 AI 的回答风格和行为"
        )
        st.session_state.system_prompt = system_prompt
        
        # 重置按钮
        if st.button("🔄 恢复默认", use_container_width=True):
            st.session_state.system_prompt = default_system_prompt
            st.rerun()
        
        # 模型参数设置
        st.divider()
        st.header("模型设置")
        model = st.selectbox(
            "选择模型",
            ["kimi-k2.5", "kimi-k2", "kimi-k1.5"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
        max_tokens = st.slider("Max Tokens", 1024, 32768, 8192, 1024)
        
        # 测试连接按钮（放在模型选择之后）
        if st.button("🔄 测试连接", use_container_width=True):
            with st.spinner("正在测试连接..."):
                try:
                    test_client = init_client()
                    if test_client:
                        # 发送一个简单的测试请求
                        response = test_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "你好"}],
                            max_tokens=10,
                            stream=False
                        )
                        st.success("✅ 连接测试成功！")
                    else:
                        st.error("❌ 客户端初始化失败")
                except Exception as e:
                    st.error(f"❌ 连接测试失败: {e}")
        
        st.divider()
        
        # 保存按钮区域
        if len(st.session_state.get("messages", [])) > 0:
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                # 覆盖保存（使用当前文件名）
                if st.button("💾 保存", use_container_width=True, help="覆盖保存到当前文件"):
                    last_user_msg = ""
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            last_user_msg = msg["content"]
                            break
                    filename = save_conversation(
                        st.session_state.messages,
                        st.session_state.get("current_file"),
                        last_user_msg,
                        force_new=False
                    )
                    st.session_state.current_file = filename
                    st.success(f"已保存: {filename}")
                    st.rerun()
            with save_col2:
                # 另存为新文件
                if st.button("💾 另存为", use_container_width=True, help="以新的问题为标题保存为新文件"):
                    last_user_msg = ""
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "user":
                            last_user_msg = msg["content"]
                            break
                    filename = save_conversation(
                        st.session_state.messages,
                        None,  # 强制生成新文件名
                        last_user_msg,
                        force_new=True
                    )
                    st.session_state.current_file = filename
                    st.success(f"已保存: {filename}")
                    st.rerun()
    
    # 检查 API Key
    client = init_client()
    if not client:
        st.error("⚠️ 请设置环境变量 MOONSHOT_API_KEY")
        st.code("set MOONSHOT_API_KEY=your_api_key", language="bash")
        return
    
    # 显示当前对话标题
    if st.session_state.current_file:
        st.caption(f"当前对话: {Path(st.session_state.current_file).stem}")
    else:
        st.caption("新对话（未保存）")
    
    # 显示历史消息
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                # 如果有 reasoning_content，以折叠形式显示
                if "reasoning_content" in msg and msg["reasoning_content"]:
                    with st.expander("💭 思考过程"):
                        st.markdown(msg["reasoning_content"])
                st.markdown(content)
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 调用大模型
        with st.chat_message("assistant"):
            reasoning_placeholder = st.empty()
            content_placeholder = st.empty()
            
            reasoning_text = ""
            content_text = ""
            
            try:
                # 构建消息列表，添加系统提示词
                messages_with_system = [
                    {"role": "system", "content": st.session_state.get("system_prompt", "")}
                ] + st.session_state.messages
                
                for chunk in stream_chat_completion(
                    client,
                    messages_with_system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if chunk["type"] == "reasoning":
                        reasoning_text += chunk["content"]
                        with reasoning_placeholder.expander("💭 思考中...", expanded=True):
                            st.markdown(reasoning_text)
                    else:
                        content_text += chunk["content"]
                        content_placeholder.markdown(content_text)
                
                # 保存 reasoning_content 到消息中
                assistant_msg = {
                    "role": "assistant",
                    "content": content_text
                }
                if reasoning_text:
                    assistant_msg["reasoning_content"] = reasoning_text
                
                st.session_state.messages.append(assistant_msg)
                
                # 重新显示思考过程（已完成的折叠形式）
                if reasoning_text:
                    reasoning_placeholder.empty()
                    with st.expander("💭 思考过程"):
                        st.markdown(reasoning_text)
                
                # 自动保存对话
                last_user_msg = prompt  # 当前用户输入
                filename = save_conversation(
                    messages=st.session_state.messages,
                    current_filename=st.session_state.get("current_file"),
                    last_user_prompt=last_user_msg,
                    force_new=False
                )
                st.session_state.current_file = filename
                st.toast(f"✅ 已自动保存", icon="💾")
                
            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    st.error("⚠️ API Key 无效或未设置，请检查 MOONSHOT_API_KEY 环境变量")
                elif "closed connection" in error_msg.lower() or "incomplete chunked" in error_msg.lower():
                    st.error("⚠️ 连接被服务器关闭，可能原因：\n"
                             "1. API Key 无效\n"
                             "2. 网络连接不稳定\n"
                             "3. 请求参数错误\n"
                             "4. 服务器暂时不可用\n\n"
                             f"详细错误: {error_msg}")
                else:
                    st.error(f"调用大模型时出错: {error_msg}")


if __name__ == "__main__":
    main()
