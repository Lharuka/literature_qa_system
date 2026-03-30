"""
文献问答系统
基于已有文献数据，通过摘要匹配找到相关文献，然后进行问答
"""

import os
import re
import json
import time
import yaml
import openai
import streamlit as st
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


# 加载配置文件
CONFIG_FILE = Path("config.yaml")
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

# API 设置
API_CONFIG = CONFIG.get("api", {})
MODEL_CONFIG = CONFIG.get("model", {})
RETRIEVAL_CONFIG = CONFIG.get("retrieval", {})
LITERATURE_CONFIG = CONFIG.get("literature", {})
SYSTEM_PROMPTS = CONFIG.get("system_prompts", {})
DATA_DIRS = CONFIG.get("data_dirs", {})

# 数据目录
DATA_DIR = Path(DATA_DIRS.get("literature_data", "literature_data"))
INDEX_FILE = DATA_DIR / "index.json"
SUMMARIES_DIR = DATA_DIR / "summaries"
TEXTS_DIR = DATA_DIR / "texts"

# 问答历史保存目录
CHAT_HISTORY_DIR = Path(DATA_DIRS.get("qa_history", "qa_history"))
CHAT_HISTORY_DIR.mkdir(exist_ok=True)




def load_literature_index() -> Dict:
    """加载文献索引"""
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_summaries() -> List[Dict]:
    """加载所有文献摘要"""
    summaries = []
    if not SUMMARIES_DIR.exists():
        return summaries
    
    for json_file in SUMMARIES_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            summaries.append(data)
    return summaries


def load_literature_text(doc_id: str) -> str:
    """加载指定文献的原文"""
    text_file = TEXTS_DIR / f"{doc_id}.txt"
    if not text_file.exists():
        return ""
    with open(text_file, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt_with_summaries(summaries: List[Dict]) -> str:
    """构建包含文献摘要的系统提示词"""
    if not summaries:
        return "你是一个文献问答助手。目前没有可用的文献数据。"
    
    prompt_parts = [
        "你是一个专业的文献问答助手。你可以根据用户的问题，从以下文献中找到相关内容并回答。",
        "",
        "=== 可用文献列表 ===",
        ""
    ]
    
    for doc in summaries:
        doc_id = doc.get("文献编码", "")
        title = doc.get("标题", "")
        abstract = doc.get("摘要", "")
        keywords = doc.get("关键字", [])
        
        prompt_parts.append(f"【文献编码】{doc_id}")
        if title:
            prompt_parts.append(f"【标题】{title}")
        prompt_parts.append(f"【摘要】{abstract}")
        prompt_parts.append(f"【关键字】{', '.join(keywords)}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        "=== 回答规则 ===",
        "1. 当用户提问时，先根据摘要判断哪些文献可能与问题相关",
        "2. 如果确定有相关文献，输出格式：",
        "   - 先说明引用了哪些文献（列出文献编码）",
        "   - 然后基于文献内容回答问题",
        "3. 如果没有相关文献，请明确告知用户",
        "4. 回答时要准确引用文献内容，不要编造信息",
        ""
    ])
    
    return "\n".join(prompt_parts)


def select_relevant_literatures(user_question: str, summaries: List[Dict], client: openai.Client, max_retries: int = None) -> List[str]:
    """使用大模型根据摘要选择相关文献，带重试机制"""
    if max_retries is None:
        max_retries = RETRIEVAL_CONFIG.get("max_retries", 3)
    
    if not summaries:
        return []
    
    # 获取配置参数
    abstract_preview = LITERATURE_CONFIG.get("abstract_preview_length", 200)
    
    # 构建选择提示词
    selection_prompt = "请根据以下文献信息，判断哪些文献与用户的问题相关。\n\n"
    selection_prompt += "=== 文献列表 ===\n\n"
    
    for doc in summaries:
        doc_id = doc.get("文献编码", "")
        title = doc.get("标题", "")
        abstract = doc.get("摘要", "")[:abstract_preview] + "..." if len(doc.get("摘要", "")) > abstract_preview else doc.get("摘要", "")
        keywords = doc.get("关键字", [])
        selection_prompt += f"文献编码: {doc_id}\n"
        if title:
            selection_prompt += f"标题: {title}\n"
        selection_prompt += f"摘要: {abstract}\n"
        selection_prompt += f"关键字: {', '.join(keywords)}\n\n"
    
    selection_prompt += f"=== 用户问题 ===\n{user_question}\n\n"
    selection_prompt += "请直接返回相关文献的编码列表（每行一个），如果没有相关文献请返回'无'。不要输出其他内容。"
    
    # 重试机制
    retry_backoff = RETRIEVAL_CONFIG.get("retry_backoff", 2)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_CONFIG.get("default_model", "kimi-k2.5"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS.get("retrieval_system", "你是一个文献检索助手，擅长根据摘要判断文献相关性。")},
                    {"role": "user", "content": selection_prompt}
                ],
                temperature=MODEL_CONFIG.get("temperature", 1.0),
                max_tokens=512
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析返回的文献编码
            if result == "无" or not result:
                return []
            
            # 提取文献编码（支持多种格式）
            doc_ids = []
            for line in result.split("\n"):
                line = line.strip()
                # 移除可能的标记符号
                line = line.strip("- []•")
                if line and len(line) == 12:  # 文献编码长度为12
                    doc_ids.append(line.upper())
            
            return doc_ids
            
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = retry_backoff ** attempt  # 指数退避
                st.warning(f"⏳ 服务器繁忙，{wait_time}秒后重试... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                st.error("❌ 服务器当前过载，请稍后再试。这是 API 服务端的临时问题，不是您的操作错误。")
                return []
        except Exception as e:
            st.error(f"❌ 文献检索失败: {e}")
            return []
    
    return []


def build_qa_prompt(user_question: str, doc_ids: List[str], uploaded_docs: set = None) -> Tuple[str, List[str], set]:
    """构建问答提示词，包含相关文献原文（避免重复上传）
    
    Args:
        user_question: 用户问题
        doc_ids: 相关文献编码列表
        uploaded_docs: 已上传过的文献编码集合
    
    Returns:
        (prompt, loaded_docs, new_uploaded_docs)
    """
    if uploaded_docs is None:
        uploaded_docs = set()
    
    if not doc_ids:
        return user_question, [], uploaded_docs
    
    # 获取配置参数
    max_text_length = LITERATURE_CONFIG.get("max_text_length", 8000)
    
    prompt_parts = [
        "=== 用户问题 ===",
        user_question,
    ]
    
    # 检查是否有新文献需要上传原文
    new_docs = [doc_id for doc_id in doc_ids if doc_id not in uploaded_docs]
    
    if new_docs:
        prompt_parts.extend(["", "=== 新上传文献原文 ===", ""])
        
        loaded_docs = []
        for doc_id in new_docs:
            text = load_literature_text(doc_id)
            if text:
                # 截取最大字符数，避免超出 token 限制
                if len(text) > max_text_length:
                    text = text[:max_text_length] + "\n...[内容已截断]"
                prompt_parts.append(f"【文献编码: {doc_id}】")
                prompt_parts.append(text)
                prompt_parts.append("")
                loaded_docs.append(doc_id)
                uploaded_docs.add(doc_id)
    
    # 如果有之前上传过的文献，只列出编码供引用
    existing_docs = [doc_id for doc_id in doc_ids if doc_id in uploaded_docs and doc_id not in new_docs]
    if existing_docs:
        prompt_parts.extend(["", "=== 已上传文献（参考之前的原文）===", ""])
        prompt_parts.append("以下文献的原文已在之前的对话中提供，如需引用请参考：")
        for doc_id in existing_docs:
            prompt_parts.append(f"- {doc_id}")
    
    prompt_parts.extend([
        "",
        "=== 回答要求 ===",
        "1. 基于文献原文回答用户问题",
        "2. 明确指出引用了哪些文献（文献编码）",
        "3. 如果文献内容不足以回答问题，请说明",
        "4. 保持回答简洁准确"
    ])
    
    return "\n".join(prompt_parts), list(uploaded_docs), uploaded_docs


def compress_literature_content(content: str) -> str:
    """将文献原文压缩为编码标记 [DOC:文献编码]
    
    匹配格式：【文献编码: XXX】\n文献内容...
    """
    import re
    # 匹配 【文献编码: XXX】 开头的内容块
    pattern = r'【文献编码:\s*([A-Z0-9]+)】\n[\s\S]*?(?=\n【文献编码:|\Z)'
    
    def replace_func(match):
        doc_id = match.group(1)
        return f"[DOC:{doc_id}]"
    
    return re.sub(pattern, replace_func, content)


def expand_literature_content(content: str) -> str:
    """将编码标记 [DOC:文献编码] 展开为文献原文"""
    pattern = r'\[DOC:([A-Z0-9]+)\]'
    
    max_text_length = LITERATURE_CONFIG.get("max_text_length", 8000)
    
    def replace_func(match):
        doc_id = match.group(1)
        text = load_literature_text(doc_id)
        if text:
            # 截取最大字符数
            if len(text) > max_text_length:
                text = text[:max_text_length] + "\n...[内容已截断]"
            return f"【文献编码: {doc_id}】\n{text}"
        return f"【文献编码: {doc_id}】\n[原文未找到]"
    
    return re.sub(pattern, replace_func, content)


def compress_messages(messages: List[Dict]) -> List[Dict]:
    """压缩消息列表中的文献原文为编码标记"""
    compressed = []
    for msg in messages:
        new_msg = msg.copy()
        if msg.get("role") == "user":
            new_msg["content"] = compress_literature_content(msg["content"])
        compressed.append(new_msg)
    return compressed


def expand_messages(messages: List[Dict]) -> List[Dict]:
    """展开消息列表中的编码标记为文献原文"""
    expanded = []
    for msg in messages:
        new_msg = msg.copy()
        if msg.get("role") == "user":
            new_msg["content"] = expand_literature_content(msg["content"])
        expanded.append(new_msg)
    return expanded


def save_qa_conversation(messages: List[Dict], selected_docs: List[str], uploaded_docs: set, 
                         current_filename: str = None, last_question: str = "", force_new: bool = False) -> str:
    """保存问答对话历史（压缩文献原文）
    
    Args:
        messages: 对话消息列表
        selected_docs: 当前对话引用的文献编码列表
        uploaded_docs: 已上传过的文献编码集合
        current_filename: 当前文件名（如果是新建对话则为None）
        last_question: 用户最后的问题，用于生成文件名
        force_new: 是否强制创建新文件（另存为）
    
    Returns:
        保存的文件名
    """
    # 获取配置参数
    title_max_length = LITERATURE_CONFIG.get("title_max_length", 20)
    
    # 过滤出 user 和 assistant 的消息
    filtered_messages = [
        msg for msg in messages 
        if msg.get("role") in ["user", "assistant"]
    ]
    
    if not filtered_messages:
        return None
    
    # 压缩文献原文
    compressed_messages = compress_messages(filtered_messages)
    
    # 只有新建对话或强制新建时，才生成新文件名
    if current_filename is None or force_new:
        # 使用用户最后的问题作为文件名前缀
        if last_question:
            safe_name = "".join(c for c in last_question if c.isalnum() or c in "_- ").strip()
            safe_name = safe_name[:title_max_length] if len(safe_name) > title_max_length else safe_name
            if not safe_name:
                safe_name = "问答记录"
        else:
            safe_name = "问答记录"
        
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename = f"{safe_name}_{timestamp}.json"
    else:
        # 使用当前文件名（覆盖保存）
        filename = current_filename
    
    filepath = CHAT_HISTORY_DIR / filename
    
    # 保存的数据包括消息（压缩后）、引用的文献和已上传的文献
    save_data = {
        "messages": compressed_messages,
        "selected_docs": selected_docs,
        "uploaded_docs": list(uploaded_docs)
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    return filename


def load_qa_conversation(filename: str) -> Tuple[List[Dict], List[str], set, str]:
    """加载问答历史并展开文献原文
    
    Returns:
        (messages, selected_docs, uploaded_docs, filename)
    """
    filepath = CHAT_HISTORY_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 兼容旧格式（只有 messages 列表）
            if isinstance(data, list):
                return data, [], set(), filename
            # 新格式（包含 messages、selected_docs 和 uploaded_docs）
            compressed_messages = data.get("messages", [])
            selected_docs = data.get("selected_docs", [])
            uploaded_docs = set(data.get("uploaded_docs", []))
            # 展开文献原文
            expanded_messages = expand_messages(compressed_messages)
            return expanded_messages, selected_docs, uploaded_docs, filename
    return [], [], set(), None


def get_qa_history_files() -> List[Path]:
    """获取所有问答历史文件列表"""
    if not CHAT_HISTORY_DIR.exists():
        return []
    files = sorted(CHAT_HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return files


def init_openai_client():
    """初始化 OpenAI 客户端"""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        return None
    return openai.Client(
        base_url=API_CONFIG.get("base_url", "https://api.moonshot.cn/v1"),
        api_key=api_key,
        timeout=API_CONFIG.get("timeout", 120.0)
    )


def stream_chat_completion(client, messages, model=None, temperature=None, max_tokens=None):
    """流式调用大模型，支持思考内容"""
    # 使用配置默认值
    if model is None:
        model = MODEL_CONFIG.get("default_model", "kimi-k2.5")
    if temperature is None:
        temperature = MODEL_CONFIG.get("temperature", 1.0)
    if max_tokens is None:
        max_tokens = MODEL_CONFIG.get("max_tokens", 8192)
    
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
        raise Exception(f"API调用失败: {str(e)}") from e


def main():
    st.set_page_config(page_title="文献问答系统", page_icon="📚", layout="wide")
    
    st.title("📚 文献问答系统")
    
    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = set()
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    # 加载文献数据
    summaries = load_all_summaries()
    literature_index = load_literature_index()
    
    # 侧边栏
    with st.sidebar:
        # 历史问答记录（放在最顶端）
        st.header("历史问答")
        history_files = get_qa_history_files()
        if history_files:
            for file in history_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    display_name = file.stem
                    if st.button(f"📄 {display_name}", key=f"load_{file.name}", use_container_width=True):
                        messages, selected_docs, uploaded_docs, filename = load_qa_conversation(file.name)
                        st.session_state.messages = messages
                        st.session_state.selected_docs = selected_docs
                        st.session_state.uploaded_docs = uploaded_docs
                        st.session_state.current_file = filename
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{file.name}"):
                        file.unlink()
                        st.rerun()
        else:
            st.info("暂无历史问答")
        
        st.divider()
        
        # 新建对话按钮
        if st.button("➕ 新建对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_docs = []
            st.session_state.uploaded_docs = set()
            st.session_state.current_file = None
            st.rerun()
        
        st.divider()
        
        st.header("文献库")
        st.info(f"当前共有 {len(summaries)} 篇文献")
        
        if summaries:
            st.subheader("文献列表")
            for doc in summaries:
                doc_id = doc.get("文献编码", "")
                title = doc.get("标题", "")
                keywords = doc.get("关键字", [])
                display_title = f"📄 {title}" if title else f"📄 {doc_id}"
                with st.expander(display_title):
                    st.caption(f"文献编码: {doc_id}")
                    st.caption("摘要")
                    st.write(doc.get("摘要", "")[:150] + "...")
                    st.caption("关键字")
                    st.write(", ".join(keywords))
        
        st.divider()
        
        # API 状态
        st.header("API 状态")
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if api_key:
            st.success("✅ API Key 已设置")
        else:
            st.error("❌ API Key 未设置")
        
        # 模型设置
        st.divider()
        st.header("模型设置")
        
        # 从配置读取默认值
        default_model = MODEL_CONFIG.get("default_model", "kimi-k2.5")
        default_temp = MODEL_CONFIG.get("temperature", 1.0)
        default_max_tokens = MODEL_CONFIG.get("max_tokens", 8192)
        
        model = st.selectbox(
            "选择模型",
            ["kimi-k2.5", "kimi-k2", "kimi-k1.5"],
            index=["kimi-k2.5", "kimi-k2", "kimi-k1.5"].index(default_model) if default_model in ["kimi-k2.5", "kimi-k2", "kimi-k1.5"] else 0
        )
        temperature = st.slider("Temperature", 0.0, 2.0, float(default_temp), 0.1)
        max_tokens = st.slider("Max Tokens", 1024, 32768, int(default_max_tokens), 1024)
        

        
        # 保存按钮区域
        st.divider()
        if len(st.session_state.get("messages", [])) > 0:
            # 提取最后一条用户消息
            last_question = ""
            for msg in reversed(st.session_state.messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if "=== 用户问题 ===" in content:
                        lines = content.split("\n")
                        for line in lines:
                            if line.strip() and not line.startswith("==="):
                                last_question = line.strip()
                                break
                    elif content.startswith("用户问题："):
                        # 处理无文献时的旧格式
                        last_question = content.replace("用户问题：", "").strip()
                    else:
                        last_question = content
                    break
            
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                # 覆盖保存（使用当前文件名）
                if st.button("💾 保存", use_container_width=True, help="覆盖保存到当前文件"):
                    filename = save_qa_conversation(
                        st.session_state.messages,
                        st.session_state.get("selected_docs", []),
                        st.session_state.get("uploaded_docs", set()),
                        st.session_state.get("current_file"),
                        last_question,
                        force_new=False
                    )
                    st.session_state.current_file = filename
                    st.success(f"✅ 已保存: {filename}")
                    st.rerun()
            with save_col2:
                # 另存为新文件
                if st.button("💾 另存为", use_container_width=True, help="保存为新文件"):
                    filename = save_qa_conversation(
                        st.session_state.messages,
                        st.session_state.get("selected_docs", []),
                        st.session_state.get("uploaded_docs", set()),
                        None,  # 强制生成新文件名
                        last_question,
                        force_new=True
                    )
                    st.session_state.current_file = filename
                    st.success(f"✅ 已保存: {filename}")
                    st.rerun()
        
        st.divider()
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.selected_docs = []
            st.session_state.uploaded_docs = set()
            st.session_state.current_file = None
            st.rerun()
    
    # 检查 API Key
    client = init_openai_client()
    if not client:
        st.error("⚠️ 请设置环境变量 MOONSHOT_API_KEY")
        st.code("set MOONSHOT_API_KEY=your_api_key", language="bash")
        return
    
    # 如果没有文献数据
    if not summaries:
        st.warning("⚠️ 没有找到文献数据，请先使用 pdf_summary.py 处理文献")
        return
    
    # 显示当前对话状态
    if st.session_state.current_file:
        st.caption(f"当前对话: {Path(st.session_state.current_file).stem}")
    else:
        st.caption("新对话（未保存）")
    
    # 显示已选择的文献
    if st.session_state.selected_docs:
        with st.expander("📎 当前对话引用的文献", expanded=False):
            for doc_id in st.session_state.selected_docs:
                doc_info = literature_index.get(doc_id, {})
                pdf_name = Path(doc_info.get("pdf_path", "")).name if doc_info else "未知"
                st.write(f"- **{doc_id}**: {pdf_name}")
    
    # 构建包含摘要的系统提示词
    system_prompt = build_system_prompt_with_summaries(summaries)
    
    # 显示历史消息
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            with st.chat_message("user"):
                # 检查是否包含文献原文（新格式）
                if "=== 用户问题 ===" in content:
                    # 提取用户问题（第一行）
                    lines = content.split("\n")
                    user_question = ""
                    for line in lines:
                        if line.strip() and not line.startswith("==="):
                            user_question = line.strip()
                            break
                    # 显示原始问题
                    st.markdown(user_question)
                    # 如果有文献内容，折叠显示
                    if "【文献编码:" in content:
                        with st.expander("📄 查看引用的文献"):
                            st.markdown(content)
                elif content.startswith("用户问题："):
                    # 处理无文献时的旧格式，去掉前缀
                    user_question = content.replace("用户问题：", "").strip()
                    st.markdown(user_question)
                else:
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
        # 第一步：选择相关文献
        with st.status("🔍 正在检索相关文献...") as status:
            doc_ids = select_relevant_literatures(prompt, summaries, client)
            if doc_ids:
                st.session_state.selected_docs = doc_ids
                status.update(label=f"✅ 找到 {len(doc_ids)} 篇相关文献: {', '.join(doc_ids)}", state="complete")
            else:
                st.session_state.selected_docs = []
                status.update(label="⚠️ 未找到相关文献，将基于通用知识回答", state="complete")
        
        # 第二步：构建包含文献原文的用户消息（避免重复上传）
        qa_prompt, loaded_docs, st.session_state.uploaded_docs = build_qa_prompt(
            prompt, doc_ids, st.session_state.uploaded_docs
        )
        
        # 添加用户消息（包含文献原文）
        st.session_state.messages.append({"role": "user", "content": qa_prompt})
        with st.chat_message("user"):
            # 只显示原始问题
            st.markdown(prompt)
        
        # 第三步：调用大模型回答
        with st.chat_message("assistant"):
            # 构建完整消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                *st.session_state.messages[:-1],  # 历史对话（不含最新用户消息）
                {"role": "user", "content": qa_prompt}
            ]
            
            # 流式输出
            reasoning_placeholder = st.empty()
            content_placeholder = st.empty()
            
            reasoning_text = ""
            content_text = ""
            
            try:
                for chunk in stream_chat_completion(
                    client,
                    messages,
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
                # 提取最后一条用户消息作为文件名前缀
                last_question = ""
                for msg in reversed(st.session_state.messages):
                    if msg.get("role") == "user":
                        # 提取原始问题（去掉文献原文部分）
                        content = msg.get("content", "")
                        if "=== 用户问题 ===" in content:
                            lines = content.split("\n")
                            for line in lines:
                                if line.strip() and not line.startswith("==="):
                                    last_question = line.strip()
                                    break
                        elif content.startswith("用户问题："):
                            # 处理无文献时的旧格式
                            last_question = content.replace("用户问题：", "").strip()
                        else:
                            last_question = content
                        break
                
                filename = save_qa_conversation(
                    st.session_state.messages,
                    st.session_state.get("selected_docs", []),
                    st.session_state.get("uploaded_docs", set()),
                    st.session_state.get("current_file"),
                    last_question,
                    force_new=False
                )
                st.session_state.current_file = filename
                st.toast(f"✅ 已自动保存", icon="💾")
                
            except Exception as e:
                st.error(f"回答生成失败: {e}")


if __name__ == "__main__":
    main()
