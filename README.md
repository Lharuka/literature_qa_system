# 文献问答系统

基于大模型的文献管理与问答系统，支持 PDF 文献摘要生成、智能问答对话、历史记录管理等功能。

## 功能特性

### 1. 文献摘要生成（pdf_summary.py）
- 自动提取 PDF 文本内容
- 调用大模型生成：标题、摘要、关键字
- 生成唯一文献编码
- 数据持久化存储

### 2. 文献问答系统（literature_qa.py）
- **智能文献检索**：根据问题自动匹配相关文献
- **上下文对话**：支持多轮对话，自动关联历史文献
- **去重机制**：避免重复上传相同文献，节省 Token
- **思考过程展示**：显示 AI 推理过程（支持 reasoning_content）
- **历史记录管理**：
  - 自动保存对话
  - 手动保存/另存为
  - 加载历史对话继续问答
  - 删除历史记录

### 3. 配置管理
- 所有超参数集中到 `config.yaml`
- 支持自定义模型、Token 限制、重试策略等

## 项目结构

```
.
├── pdf_summary.py          # 文献摘要生成工具
├── literature_qa.py        # 文献问答系统主程序
├── config.yaml             # 配置文件
├── README.md               # 项目说明
├── literature_data/        # 文献数据目录
│   ├── summaries/          # 文献摘要 JSON
│   ├── texts/              # 提取的文本文件
│   └── index.json          # 文献索引
└── qa_history/             # 问答历史记录
    └── *.json              # 对话记录文件
```

## 安装依赖

```bash
pip install pymupdf openai streamlit pyyaml
```

## 使用方法

### 1. 设置 API Key

```powershell
# Windows PowerShell
$env:MOONSHOT_API_KEY="your_api_key_here"

# Linux/Mac
export MOONSHOT_API_KEY="your_api_key_here"
```

### 2. 生成文献摘要

```bash
python pdf_summary.py
```

按提示输入 PDF 文件路径，系统将自动：
1. 提取 PDF 文本
2. 调用大模型生成标题、摘要、关键字
3. 保存到 `literature_data/` 目录

### 3. 启动问答系统

```bash
streamlit run literature_qa.py
```

浏览器将自动打开界面，您可以：
- 查看文献库列表
- 输入问题开始对话
- 查看历史问答记录
- 管理对话保存/加载

## 配置说明

编辑 `config.yaml` 自定义参数：

```yaml
# API 设置
api:
  base_url: "https://api.moonshot.cn/v1"
  timeout: 120.0

# 模型设置
model:
  default_model: "kimi-k2.5"  # 可选: kimi-k2.5, kimi-k2, kimi-k1.5
  temperature: 1.0
  max_tokens: 8192

# 文献检索设置
retrieval:
  max_retries: 3              # 检索失败重试次数
  retry_backoff: 2            # 重试间隔基数

# 文献内容设置
literature:
  max_text_length: 8000       # 每篇文献上传最大字符数
  abstract_preview_length: 200 # 侧边栏摘要预览长度
  title_max_length: 20        # 文件名问题长度限制

# 数据目录
data_dirs:
  literature_data: "literature_data"
  qa_history: "qa_history"
```

## 工作流程

### 文献处理流程
```
PDF 文件 → 文本提取 → 大模型生成摘要 → 保存到 literature_data/
```

### 问答流程
```
用户提问 → 检索相关文献 → 上传原文（去重）→ 大模型回答 → 自动保存
```

### 数据压缩机制
- **保存时**：文献原文 `[DOC:编码]` 压缩为标记
- **加载时**：标记自动展开为原文
- **节省空间**：避免重复存储大段文本

## 界面说明

### 侧边栏
- **历史问答**：查看/加载/删除历史对话
- **新建对话**：开始新的问答会话
- **文献库**：查看所有可用文献
- **模型设置**：调整 Temperature、Max Tokens 等参数
- **保存按钮**：手动保存/另存为

### 主界面
- 显示当前对话状态（文件名或"新对话"）
- 显示引用的文献列表
- 用户消息只显示问题，文献原文折叠显示
- AI 回答显示思考过程（可折叠）

## 注意事项

1. **API Key 安全**：不要泄露 `MOONSHOT_API_KEY`
2. **Token 消耗**：上传长文献会消耗较多 Token，建议合理设置 `max_text_length`
3. **自动保存**：只在 AI 成功回答后触发，失败时对话不会保存
4. **历史记录**：保存在 `qa_history/` 目录，可手动备份
5. **数据兼容性**：加载旧格式历史记录时部分功能可能受限

## 常见问题

### Q: 为什么对话没有自动保存？
A: 自动保存只在 AI 成功回答后触发。如果 API 报错（429限流、网络错误等），对话不会保存。

### Q: 如何避免重复上传文献？
A: 系统会自动跟踪已上传的文献，同一篇文献在后续对话中只会显示编码引用，不会重复上传原文。

### Q: 可以修改已保存的对话吗？
A: 可以加载历史对话继续问答，然后点击"保存"覆盖原文件，或"另存为"创建新文件。

### Q: 支持哪些 PDF？
A: 支持普通文本 PDF。扫描版 PDF 需要 OCR 功能（当前版本暂未支持）。

## 技术栈

- **PDF 处理**：PyMuPDF
- **大模型 API**：OpenAI SDK (Moonshot API)
- **Web 界面**：Streamlit
- **数据格式**：JSON
- **配置管理**：YAML

## License

MIT License
