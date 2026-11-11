#!/usr/bin/env python3
"""
硅基流动 OCR API 调用脚本
支持多种模型：PaddleOCR-VL、DeepSeek-OCR、Qwen2.5-VL、GLM-4.6V 等
支持文件类型：图片(JPG/PNG/WEBP等)、PDF

使用示例：
    python siliconflow_ocr.py --file example.png
    python siliconflow_ocr.py --file document.pdf --model deepseek-ai/deepseek-ocr
    python siliconflow_ocr.py --file image.jpg --prompt "提取所有文字"
"""

import argparse
import base64
import io
import os
import sys
from pathlib import Path

from PIL import Image
from openai import OpenAI


# 支持的 OCR/视觉模型列表
SUPPORTED_MODELS = {
    "paddleocr-vl": "PaddlePaddle/PaddleOCR-VL",
    "paddleocr-vl-1.5": "PaddlePaddle/PaddleOCR-VL-1.5",
    "deepseek-ocr": "deepseek-ai/deepseek-ocr",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "glm-4v": "THUDM/glm-4v-9b",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B",
}

# 默认提示词
DEFAULT_PROMPTS = {
    "default": "请识别图片中的文字内容，保持原有排版格式。",
    "markdown": "请将文档转换为 Markdown 格式输出。",
    "structured": "请提取图片中的所有文字信息，并以结构化方式返回。",
}


def get_api_key():
    """从环境变量获取 API Key"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 SILICONFLOW_API_KEY")
        print("获取方式：登录 https://cloud.siliconflow.cn 创建 API Key")
        sys.exit(1)
    return api_key


def convert_image_to_base64(image_path: str, format: str = "WEBP", quality: int = 85) -> str:
    """
    将图片转换为 base64 编码
    
    Args:
        image_path: 图片文件路径
        format: 输出格式，默认为 WEBP（体积小）
        quality: 图片质量，默认 85
    
    Returns:
        base64 编码的字符串
    """
    try:
        with Image.open(image_path) as img:
            # 如果是RGBA模式，转换为RGB（避免某些格式问题）
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存到字节流
            byte_arr = io.BytesIO()
            img.save(byte_arr, format=format, quality=quality)
            byte_arr = byte_arr.getvalue()
            
            return base64.b64encode(byte_arr).decode('utf-8')
    except Exception as e:
        print(f"图片转换错误: {e}")
        return None


def read_pdf_as_base64(pdf_path: str) -> str:
    """
    读取 PDF 文件并转为 base64 编码
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        base64 编码的字符串
    """
    try:
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"PDF 读取错误: {e}")
        return None


def recognize_image(
    client: OpenAI,
    model: str,
    base64_data: str,
    prompt: str,
    mime_type: str = "image/webp",
    stream: bool = False,
    max_tokens: int = 4000
) -> str:
    """
    调用硅基流动 API 进行图像识别
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        base64_data: base64 编码的图片数据
        prompt: 提示词
        mime_type: MIME 类型
        stream: 是否流式输出
        max_tokens: 最大输出 token 数
    
    Returns:
        识别结果文本
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            stream=stream,
            max_tokens=max_tokens
        )
        
        if stream:
            # 流式输出
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end='', flush=True)
                    full_response += text_chunk
            return full_response
        else:
            # 非流式输出
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"API 调用错误: {e}")
        return None


def recognize_pdf(
    client: OpenAI,
    model: str,
    base64_data: str,
    prompt: str,
    stream: bool = False,
    max_tokens: int = 4000
) -> str:
    """
    调用硅基流动 API 进行 PDF 识别（仅 DeepSeek-OCR 支持）
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        base64_data: base64 编码的 PDF 数据
        prompt: 提示词
        stream: 是否流式输出
        max_tokens: 最大输出 token 数
    
    Returns:
        识别结果文本
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{base64_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            stream=stream,
            max_tokens=max_tokens
        )
        
        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end='', flush=True)
                    full_response += text_chunk
            return full_response
        else:
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"API 调用错误: {e}")
        return None


def save_result(result: str, output_path: str, input_file: str = None):
    """
    保存识别结果到文件
    
    Args:
        result: 识别结果文本
        output_path: 输出文件路径
        input_file: 输入文件路径（用于默认命名）
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n结果已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def generate_output_filename(input_path: Path, model: str) -> str:
    """
    根据输入文件名自动生成输出文件名
    
    Args:
        input_path: 输入文件路径
        model: 使用的模型名称
    
    Returns:
        输出文件路径
    """
    stem = input_path.stem
    parent = input_path.parent
    output_name = f"{stem}_{model}_result.txt"
    return str(parent / output_name) if parent else output_name


def main():
    parser = argparse.ArgumentParser(
        description="硅基流动 OCR 工具 - 识别图片/PDF 中的文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -f image.png                              # 识别图片
  %(prog)s -f image.png -o result.txt                # 保存结果到指定文件
  %(prog)s -f doc.pdf -m deepseek-ocr                # 识别 PDF
  %(prog)s -f img.jpg -p "提取表格数据"               # 自定义提示词
  %(prog)s -f img.png -s                             # 流式输出
  %(prog)s -f img.png --auto-save                    # 自动保存结果

环境变量:
  SILICONFLOW_API_KEY    必需，从 https://cloud.siliconflow.cn 获取
        """
    )
    
    parser.add_argument("-f", "--file", required=True, help="要识别的文件路径（图片或 PDF）")
    parser.add_argument(
        "-m", "--model", 
        default="paddleocr-vl",
        choices=list(SUPPORTED_MODELS.keys()),
        help="使用的模型（默认: paddleocr-vl）"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="default",
        help="识别提示词（或使用预设: default/markdown/structured）"
    )
    parser.add_argument(
        "-s", "--stream",
        action="store_true",
        help="启用流式输出"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出文件路径（默认: 屏幕输出）"
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="自动保存结果到文件（文件名: 输入文件名_模型名_result.txt）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="最大输出 token 数（默认: 4000）"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有支持的模型"
    )
    
    args = parser.parse_args()
    
    # 列出支持的模型
    if args.list_models:
        print("支持的 OCR/视觉模型:")
        for key, value in SUPPORTED_MODELS.items():
            print(f"  {key}: {value}")
        return
    
    # 获取文件路径
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"错误：文件不存在: {file_path}")
        sys.exit(1)
    
    # 获取 API Key 并初始化客户端
    api_key = get_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # 获取模型完整名称
    model_name = SUPPORTED_MODELS.get(args.model, args.model)
    
    # 处理提示词
    if args.prompt in DEFAULT_PROMPTS:
        prompt = DEFAULT_PROMPTS[args.prompt]
    else:
        prompt = args.prompt
    
    # 根据文件类型处理
    file_ext = file_path.suffix.lower()
    
    if file_ext in ['.pdf']:
        # PDF 文件处理
        if args.model != "deepseek-ocr":
            print("提示：PDF 识别建议使用 deepseek-ocr 模型，其他模型可能不支持")
        
        print(f"正在识别 PDF: {file_path.name}")
        print(f"使用模型: {model_name}")
        print("-" * 50)
        
        base64_data = read_pdf_as_base64(str(file_path))
        if not base64_data:
            sys.exit(1)
        
        result = recognize_pdf(
            client, model_name, base64_data, prompt,
            stream=args.stream, max_tokens=args.max_tokens
        )
        
    elif file_ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff']:
        # 图片文件处理
        print(f"正在识别图片: {file_path.name}")
        print(f"使用模型: {model_name}")
        print("-" * 50)
        
        # 根据原始格式选择 MIME 类型
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif',
            '.tiff': 'image/tiff',
        }
        mime_type = mime_types.get(file_ext, 'image/webp')
        
        # 转换为 base64（统一转为 WEBP 减小体积）
        base64_data = convert_image_to_base64(str(file_path))
        if not base64_data:
            sys.exit(1)
        
        result = recognize_image(
            client, model_name, base64_data, prompt,
            mime_type="image/webp",
            stream=args.stream, max_tokens=args.max_tokens
        )
        
    else:
        print(f"不支持的文件类型: {file_ext}")
        print("支持的格式: .pdf, .jpg, .jpeg, .png, .webp, .bmp, .gif, .tiff")
        sys.exit(1)
    
    # 输出结果
    if result:
        # 保存到文件
        if args.output:
            save_result(result, args.output)
        elif args.auto_save:
            output_path = generate_output_filename(file_path, args.model)
            save_result(result, output_path)
        
        # 屏幕输出（非流式模式下）
        if not args.stream:
            print("\n识别结果:")
            print("=" * 50)
            print(result)
            print("=" * 50)
    else:
        print("识别失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
