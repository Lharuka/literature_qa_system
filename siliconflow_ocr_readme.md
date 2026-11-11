# 硅基流动 OCR 脚本使用说明

## 安装依赖

```bash
pip install openai Pillow
```

## 配置 API Key

获取 API Key：https://cloud.siliconflow.cn/account/ak

```bash
# Windows PowerShell
$env:SILICONFLOW_API_KEY="your-api-key-here"

# Windows CMD
set SILICONFLOW_API_KEY=your-api-key-here

# Linux/Mac
export SILICONFLOW_API_KEY="your-api-key-here"
```

## 基本用法

### 1. 识别图片

```bash
# 使用默认模型 (PaddleOCR-VL) 识别图片
python siliconflow_ocr.py -f image.png

# 使用 Qwen2.5-VL 模型
python siliconflow_ocr.py -f image.jpg -m qwen2.5-vl-72b

# 使用 DeepSeek-OCR 模型
python siliconflow_ocr.py -f photo.png -m deepseek-ocr
```

### 2. 识别 PDF

```bash
# PDF 识别需要使用 DeepSeek-OCR 模型
python siliconflow_ocr.py -f document.pdf -m deepseek-ocr
```

### 3. 自定义提示词

```bash
# 使用自定义提示词
python siliconflow_ocr.py -f invoice.png -p "提取发票号码、金额和日期"

# 使用预设提示词（转换为 Markdown）
python siliconflow_ocr.py -f doc.jpg -p markdown

# 使用预设提示词（结构化提取）
python siliconflow_ocr.py -f table.png -p structured
```

### 4. 保存结果到文件

```bash
# 保存到指定文件
python siliconflow_ocr.py -f image.png -o result.txt

# 保存为 Markdown 格式
python siliconflow_ocr.py -f doc.pdf -m deepseek-ocr -p markdown -o doc.md

# 自动命名保存（文件名格式: 原文件名_模型名_result.txt）
python siliconflow_ocr.py -f image.png --auto-save
# 将生成: image_paddleocr-vl_result.txt
```

### 5. 流式输出

```bash
# 实时显示识别结果（屏幕显示，可配合 --auto-save 保存）
python siliconflow_ocr.py -f large_image.png -s --auto-save
```

### 6. 查看支持的模型

```bash
python siliconflow_ocr.py --list-models
```

## 支持的模型

| 模型别名 | 完整名称 | 特点 |
|---------|---------|------|
| paddleocr-vl | PaddlePaddle/PaddleOCR-VL | 百度开源，中文识别效果好 |
| paddleocr-vl-1.5 | PaddlePaddle/PaddleOCR-VL-1.5 | 新版 PaddleOCR |
| deepseek-ocr | deepseek-ai/deepseek-ocr | 支持 PDF 输入 |
| qwen2.5-vl-72b | Qwen/Qwen2.5-VL-72B-Instruct | 阿里大模型，能力强 |
| qwen2.5-vl-32b | Qwen/Qwen2.5-VL-32B-Instruct | 速度与效果平衡 |
| qwen2.5-vl-7b | Qwen/Qwen2.5-VL-7B-Instruct | 轻量快速 |
| glm-4v | THUDM/glm-4v-9b | 智谱 GLM 视觉模型 |
| qwen3-vl-8b | Qwen/Qwen3-VL-8B | Qwen3 系列 |

## Python API 调用示例

```python
import os
from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 读取图片并转 base64
with open("image.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

# 调用 API
response = client.chat.completions.create(
    model="PaddlePaddle/PaddleOCR-VL",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "请识别图片中的文字内容"
                }
            ]
        }
    ],
    max_tokens=4000
)

print(response.choices[0].message.content)
```

## 注意事项

1. **API Key**：必须设置环境变量 `SILICONFLOW_API_KEY`
2. **PDF 识别**：只有 `deepseek-ocr` 模型支持 PDF 文件
3. **图片大小**：建议图片不超过 4MB，脚本会自动压缩为 WEBP 格式
4. **免费额度**：硅基流动提供免费 Token 额度，具体查看官网
