"""
PDF 文献摘要生成工具
使用 OCR 提取 PDF 文本，调用大模型生成摘要和关键字
"""

import os
import json
import hashlib
import openai
from pathlib import Path
from datetime import datetime


# 数据存储目录
DATA_DIR = Path("literature_data")
TEXT_DIR = DATA_DIR / "texts"
JSON_DIR = DATA_DIR / "summaries"
INDEX_FILE = DATA_DIR / "index.json"


def init_directories():
    """初始化存储目录"""
    DATA_DIR.mkdir(exist_ok=True)
    TEXT_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)


def generate_doc_id(pdf_path: str) -> str:
    """生成文献唯一编码：基于文件路径和修改时间的哈希"""
    file_stat = Path(pdf_path).stat()
    content = f"{pdf_path}_{file_stat.st_mtime}_{file_stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()[:12].upper()


def extract_text_from_pdf(pdf_path: str) -> str:
    """使用 PyMuPDF 提取 PDF 文本"""
    import fitz
    
    text_content = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text_content.append(f"=== 第 {page_num + 1} 页 ===\n{text}")
    
    doc.close()
    return "\n\n".join(text_content)


def save_text_file(doc_id: str, text_content: str) -> Path:
    """保存提取的文本文件"""
    text_file = TEXT_DIR / f"{doc_id}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text_content)
    return text_file


def init_openai_client():
    """初始化 OpenAI 客户端"""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 MOONSHOT_API_KEY")
    return openai.Client(
        base_url="https://api.moonshot.cn/v1",
        api_key=api_key,
        timeout=120.0
    )


def generate_summary(client: openai.Client, text_content: str) -> dict:
    """调用大模型生成标题、摘要和关键字"""
    
    # 如果文本太长，截取前 100000 字（约 40000 个汉字）
    max_chars = 100000
    if len(text_content) > max_chars:
        text_content = text_content[:max_chars] + "\n...[内容过长，已截断]"
    
    prompt = f"""请阅读以下文献内容，生成标题、摘要和关键字。

要求：
1. 标题：拟定一个简短的文献标题（不超过20个字），能概括文献主题
2. 摘要：200-300字，概括文献的核心内容、研究方法和主要结论
3. 关键字：提取3-5个核心概念，以列表形式返回

请严格按照以下 JSON 格式返回，不要添加任何其他内容：
{{
    "标题": "文献标题",
    "摘要": "摘要内容",
    "关键字": ["关键词1", "关键词2", "关键词3"]
}}

文献内容：
{text_content}
"""

    response = client.chat.completions.create(
        model="kimi-k2.5",
        messages=[
            {"role": "system", "content": "你是一个专业的学术文献分析助手，擅长提取文献的核心内容和关键字。"},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_tokens=2048
    )
    
    result_text = response.choices[0].message.content
    
    # 解析 JSON 结果
    try:
        # 尝试直接解析
        result = json.loads(result_text)
    except json.JSONDecodeError:
        # 如果解析失败，尝试提取 JSON 部分
        import re
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError(f"无法解析模型返回的结果: {result_text}")
    
    return result


def save_summary_json(doc_id: str, summary_data: dict, pdf_path: str) -> Path:
    """保存摘要 JSON 文件"""
    output_data = {
        "文献编码": doc_id,
        "标题": summary_data.get("标题", ""),
        "摘要": summary_data.get("摘要", ""),
        "关键字": summary_data.get("关键字", [])
    }
    
    json_file = JSON_DIR / f"{doc_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return json_file


def update_index(doc_id: str, pdf_path: str, text_file: Path, json_file: Path):
    """更新索引文件，关联文献编码和文件"""
    index = {}
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            index = json.load(f)
    
    index[doc_id] = {
        "pdf_path": str(Path(pdf_path).absolute()),
        "text_file": str(text_file.absolute()),
        "json_file": str(json_file.absolute()),
        "created_at": datetime.now().isoformat()
    }
    
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def process_pdf(pdf_path: str):
    """处理单个 PDF 文件的主流程"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"❌ 文件不存在: {pdf_path}")
        return False
    
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"❌ 不是 PDF 文件: {pdf_path}")
        return False
    
    print(f"\n📄 处理文件: {pdf_path.name}")
    print("-" * 50)
    
    # 1. 生成文献编码
    doc_id = generate_doc_id(str(pdf_path))
    print(f"🔑 文献编码: {doc_id}")
    
    # 检查是否已经处理过
    existing_json = JSON_DIR / f"{doc_id}.json"
    if existing_json.exists():
        print(f"⚠️ 该文献已处理过，跳过")
        print(f"   摘要文件: {existing_json}")
        return True
    
    # 2. 提取文本
    print("🔍 正在提取文本...")
    try:
        text_content = extract_text_from_pdf(str(pdf_path))
        text_file = save_text_file(doc_id, text_content)
        print(f"✅ 文本已保存: {text_file}")
    except Exception as e:
        print(f"❌ 文本提取失败: {e}")
        return False
    
    # 3. 调用大模型生成摘要
    print("🤖 正在生成摘要...")
    try:
        client = init_openai_client()
        summary_data = generate_summary(client, text_content)
        print(f"✅ 摘要生成完成")
    except Exception as e:
        print(f"❌ 摘要生成失败: {e}")
        return False
    
    # 4. 保存 JSON
    json_file = save_summary_json(doc_id, summary_data, str(pdf_path))
    print(f"✅ 摘要已保存: {json_file}")
    
    # 5. 更新索引
    update_index(doc_id, str(pdf_path), text_file, json_file)
    print(f"✅ 索引已更新")
    
    # 显示结果
    print("\n📋 生成结果:")
    print("-" * 50)
    print(f"文献编码: {doc_id}")
    print(f"标题: {summary_data.get('标题', '')}")
    print(f"摘要: {summary_data.get('摘要', '')[:100]}...")
    print(f"关键字: {', '.join(summary_data.get('关键字', []))}")
    print("-" * 50)
    
    return True


def main():
    """命令行主函数"""
    init_directories()
    
    print("=" * 60)
    print("📚 PDF 文献摘要生成工具")
    print("=" * 60)
    print(f"数据存储目录: {DATA_DIR.absolute()}")
    print(f"文本文件目录: {TEXT_DIR.absolute()}")
    print(f"摘要文件目录: {JSON_DIR.absolute()}")
    print("-" * 60)
    
    while True:
        print("\n请输入 PDF 文件路径（输入 'q' 退出）:")
        user_input = input("> ").strip()
        
        if user_input.lower() in ["q", "quit", "exit"]:
            print("👋 再见！")
            break
        
        if not user_input:
            continue
        
        # 处理输入路径（支持带引号的路径）
        pdf_path = user_input.strip('"\'')
        
        # 处理文件
        success = process_pdf(pdf_path)
        
        if success:
            print("\n✨ 处理完成！")
        else:
            print("\n❌ 处理失败，请检查错误信息。")


if __name__ == "__main__":
    main()
