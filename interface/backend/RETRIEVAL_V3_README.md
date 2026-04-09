# 图表检索 V3 版本更新说明

## 概述

此更新将原有的图表检索方式升级为 V3 版本，基于 `evaluate_5types_human_v3.py` 的核心逻辑实现。

## 主要变化

### 1. 模型更新
- **基础模型**: `BAAI/BGE-VL-base`
- **Checkpoint**: `/mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/retrieval_training/output_4types/best_model.pt`
- **模型类型**: `MultiHeadBGEVL` - 支持5个aspect的多头特征提取

### 2. 检索方式更新
- **5-Aspect分解检索**: 支持将查询分解为 content, style, layout, illustration, chart_type 五个维度
- **LLM Query Rewrite**: 使用 GPT-5-mini 或 Qwen 模型对查询进行重写和分解
- **Binary Weights**: 支持二进制权重模式，简化权重计算

### 3. 配置文件更新 (`api_server.py`)

```python
app_config = {
    # V3 Retrieval Configuration (New)
    'base_model': 'BAAI/BGE-VL-base',
    'checkpoint': '/mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/retrieval_training/output_4types/best_model.pt',
    'data_root': '/mnt/share/public/converted/converted',
    'split_file': str(CHART_RETRIEVAL_DIR / "retrieval_training/data/train_filtered_thresh_0.92.json"),
    'metadata_file': '/mnt/share/xujing/Chart_Retrieval/VizRetrieval/data/samples_info_200k_new.json',
    'hierarchy_file': str(CHART_RETRIEVAL_DIR / "data_processing/chart_types_hierarchy.json"),
    'llm_model': 'gpt-5-mini-2025-08-07',
    'use_binary_weights': True,
    'rewrite_query': True,  # 是否使用LLM rewrite
}
```

## 启动方式

### 使用启动脚本（推荐）

```bash
cd /mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/interface/backend
./start_server.sh
```

### 手动启动

```bash
# 激活conda环境
source /home/xujing/miniconda3/etc/profile.d/conda.sh
conda activate qwen

# 设置环境变量
export HF_HOME="/mnt/share/xujing/hf"
export HF_ENDPOINT="https://hf-mirror.com"

# 启动服务
cd /mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/interface/backend
python api_server.py
```

## API 使用

API端点保持不变，与之前版本兼容：

### 1. 聊天检索
```bash
POST /api/chat
Content-Type: multipart/form-data

Parameters:
- user_text: 用户输入文本
- session_id: 会话ID
- model_name: 模型名称 (默认: Qwen/Qwen2.5-VL-72B-Instruct)
- user_image: 用户上传的图片 (可选)
```

### 2. 精炼检索
```bash
POST /api/chat/refine
Content-Type: multipart/form-data

Parameters:
- session_id: 会话ID
- previous_query: 上一次的检索查询
- refinement_text: 用户的精炼指令
- model_name: 模型名称
- user_image: 用户上传的图片 (可选)
```

### 3. 生成最终回答
```bash
POST /api/chat/finalize
Content-Type: multipart/form-data

Parameters:
- user_text: 用户输入文本
- session_id: 会话ID
- selected_images: 用户选择的图片路径列表 (JSON格式)
- selection_mode: 选择模式 ("auto" 或 "manual")
- model_name: 模型名称
- user_image: 用户上传的图片 (可选)
```

## 新特性说明

### LLM Query Rewrite
当 `rewrite_query=True` 时，系统会自动使用LLM将用户查询分解为5个aspect：
- **content**: 数据内容、主题、实体
- **style**: 视觉风格、颜色、艺术风格
- **layout**: 布局结构、排列方式
- **illustration**: 装饰元素、图标
- **chart_type**: 图表类型（Bar Chart, Line Chart等）

### Chart Type 过滤
系统现在支持基于chart type的相似度计算，并可以利用LLM建议的chart types进行过滤。

### Binary Weights
当 `use_binary_weights=True` 时，系统使用二进制权重（0或1），忽略具体的权重值，简化检索逻辑。

## 文件结构

```
interface/backend/
├── api_server.py           # 主API服务器（已更新）
├── retrieval_v3.py         # 新的V3检索模块
├── mllm.py                 # MLLM处理类
├── start_server.sh         # 启动脚本
├── requirements.txt        # 依赖项（已更新）
└── RETRIEVAL_V3_README.md  # 本说明文件
```

## 依赖项

确保已安装以下依赖（已添加到requirements.txt）：

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pillow>=9.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
openai>=1.0.0
```

## 故障排除

### 1. CUDA内存不足
如果遇到CUDA OOM错误，可以尝试：
- 减少batch_size（在retrieval_v3.py中修改）
- 使用CPU进行检索（修改device参数）

### 2. Gallery加载慢 (已优化)
**重要更新**: 现在 gallery 特征采用延迟加载（lazy loading）策略：
- 服务器启动时不会立即加载 gallery 特征
- 第一次搜索请求时会触发 gallery 特征加载
- 这可能导致第一次搜索请求耗时较长（几分钟）
- 后续搜索请求将正常快速响应

可以通过健康检查端点查看 gallery 加载状态：
```bash
GET /api/health
```

返回：
```json
{
  "status": "healthy",
  "message": "MLLM API is running",
  "retrieval_ready": true,  // 表示 gallery 已加载完成
  "retriever_type": "v3"
}
```

### 3. LLM API调用失败
检查API密钥和网络连接。系统会在LLM调用失败时自动回退到原始查询（仅使用content aspect）。

## 回滚到旧版本

如需回滚到旧版本，修改 `api_server.py`：

```python
# 注释掉V3导入
# from retrieval_v3 import MultimodalRetrieverV3, ASPECTS, CHART_TYPE_ASPECT

# 恢复旧版本导入
from src.retrieval.multimodal_retriever import MultimodalRetriever

# 恢复旧版本初始化
retriever = MultimodalRetriever(
    embeddings_path=Path(app_config['base_dir']) / "embeddings" / "image_embeddings.npy", 
    metadata_path=Path(app_config['base_dir']) / "metadata" / "image_metadata.json",
    model_type=app_config['model_type'],
    model_path=app_config['model_path']
)
```

## 参考

- `evaluate_5types_human_v3.py`: V3评估脚本
- `evaluate_5types_human_v2.py`: V2评估脚本（包含LLM rewrite函数）
- `evaluate_5types_human.py`: 基础评估脚本（包含模型定义）
- `run_eval_5types_human_v3.sh`: V3运行脚本参考
