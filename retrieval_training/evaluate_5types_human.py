import os
import json
import logging
import torch
import argparse
import datetime
import time
import re
import math
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===========================
# 1. Model Definition
# ===========================

ASPECTS = ['content', 'style', 'layout', 'illustration']
CHART_TYPE_ASPECT = 'chart_type'

# 预定义的 Chart Type 集合 (对应 Hierarchy 中的 Roots + 重要的 Level 2)
# 基于 chart_types_hierarchy.json 的 roots
CHART_TYPE_POOL = [
    "Bar Chart", "Line Chart", "Area Chart", "Radar Chart", "Pie Chart",
    "Scatterplot", "Gauge Chart", "Treemap", "Diagram", "Histogram",
    "Range Chart", "Funnel Chart", "Pyramid Chart"
]

# 专家定义的图表类型相似度矩阵 (Soft Match)
# 解决细粒度分类混淆和视觉相似性问题
# 未列出的组合默认为 0.0
CHART_TYPE_SIMILARITY = {
    frozenset(["Bar Chart", "Histogram"]): 0.9,      # 分布 vs 比较，但视觉极像
    frozenset(["Bar Chart", "Funnel Chart"]): 0.6,   # Funnel 是中心对齐的 Bar
    frozenset(["Bar Chart", "Pyramid Chart"]): 0.6,  # Pyramid 也是 Bar 的变体
    frozenset(["Line Chart", "Area Chart"]): 0.8,    # Area 是填充的 Line (Increased from 0.7)
    frozenset(["Line Chart", "Scatterplot"]): 0.4,   # 趋势 vs 散点，有时视觉重叠
    frozenset(["Line Chart", "Radar Chart"]): 0.3,   # Radar 是极坐标 Line
    frozenset(["Area Chart", "Radar Chart"]): 0.3,   # 填充 Radar 像 Area
    frozenset(["Area Chart", "Scatterplot"]): 0.6,   # 为了覆盖 "Proportional Area Chart (Circle)" 这种像 Bubble 的情况
    frozenset(["Pie Chart", "Gauge Chart"]): 0.7,    # 都是圆形，表达比例/进度 (Increased from 0.5)
    frozenset(["Pie Chart", "Diagram"]): 0.2,        # 有时圆形图会被归为 Diagram
    frozenset(["Funnel Chart", "Pyramid Chart"]): 0.8 # 形状倒置关系
}

PREFIXES = {
    'content': '[CONTENT] ',
    'style': '[STYLE] ',
    'layout': '[LAYOUT] ',
    'illustration': '[ILLUSTRATION] '
}

class MultiHeadBGEVL(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        logger.info(f"Initializing MultiHeadBGEVL with base: {base_model_name}")
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

        if hasattr(self.base_model, 'visual_projection'):
            vis_in = self.base_model.visual_projection.in_features
            vis_out = self.base_model.visual_projection.out_features
            original_vis_weight = self.base_model.visual_projection.weight.data
        else:
            raise ValueError("Base model does not have 'visual_projection'")

        if hasattr(self.base_model, 'text_projection'):
            txt_in = self.base_model.text_projection.in_features
            txt_out = self.base_model.text_projection.out_features
            original_txt_weight = self.base_model.text_projection.weight.data
        else:
            raise ValueError("Base model does not have 'text_projection'")

        self.visual_heads = nn.ModuleDict()
        self.text_heads = nn.ModuleDict()

        for aspect in ASPECTS:
            # Init with original weights to ensure baseline performance matches base model
            v_head = nn.Linear(vis_in, vis_out, bias=False)
            v_head.weight.data.copy_(original_vis_weight)
            self.visual_heads[aspect] = v_head

            t_head = nn.Linear(txt_in, txt_out, bias=False)
            t_head.weight.data.copy_(original_txt_weight)
            self.text_heads[aspect] = t_head

        if hasattr(self.base_model, 'processor'):
            self.processor = self.base_model.processor
        elif hasattr(self.base_model, 'set_processor'):
            self.base_model.set_processor(base_model_name)
            self.processor = self.base_model.processor

    def forward_image_features(self, image_inputs):
        """Extract 4 image features for each aspect"""
        vision_outputs = self.base_model.vision_model(image_inputs)
        image_pooler_output = vision_outputs[1]

        embeds = {}
        for aspect in ASPECTS:
            proj = self.visual_heads[aspect](image_pooler_output)
            embeds[aspect] = F.normalize(proj, p=2, dim=-1)
        return embeds

    def forward_text_features(self, text_inputs, aspect):
        """Extract text features for a specific aspect"""
        text_outputs = self.base_model.text_model(text_inputs)
        text_pooler_output = text_outputs[1]

        proj = self.text_heads[aspect](text_pooler_output)
        return F.normalize(proj, p=2, dim=-1)

# ===========================
# 2. Data Loading & Metadata Hierarchy
# ===========================

class ChartTypeMapper:
    """Helper to map fine-grained chart types to coarse-grained ROOT types"""
    def __init__(self, hierarchy_path):
        self.mapping = {} # specific_type -> root_type
        self.father_mapping = {} # specific_type -> father_type
        self._load_hierarchy(hierarchy_path)

    def _load_hierarchy(self, path):
        if not os.path.exists(path):
            logger.warning(f"Hierarchy file not found: {path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Strategy 1: Iterate 'flat_mapping' if exists (most robust)
        if 'flat_mapping' in data:
            for specific_type, details in data['flat_mapping'].items():
                # 'path' is usually [Root, Level2, ...]
                # We want the Root (index 0)
                if details.get('path'):
                    root_type = details['path'][0]
                    if len(details['path']) == 1:
                        father_type = root_type
                    else:
                        father_type = details['path'][1]
                    self.mapping[specific_type.lower()] = root_type
                    self.mapping[specific_type] = root_type # keep original case too
                    self.father_mapping[specific_type] = father_type
                    self.father_mapping[specific_type.lower()] = father_type

        # Strategy 2: If flat_mapping missing, traverse 'roots'
        # (Assuming 'flat_mapping' exists based on provided file content)

        # Also map Roots to themselves
        for root in CHART_TYPE_POOL:
            self.mapping[root.lower()] = root
            self.mapping[root] = root

    def get_root_type(self, specific_type):
        if not specific_type:
            return None
        # Normalize
        st_clean = specific_type.strip()
        # Try exact match first
        if st_clean in self.mapping:
            return self.mapping[st_clean]
        if st_clean.lower() in self.mapping:
            return self.mapping[st_clean.lower()]

        # Try substring match as fallback?
        # (e.g., "Horizontal Bar Chart" might not be in mapping but "Bar Chart" is root)
        # Actually based on user file, "Horizontal Bar Chart" IS in flat_mapping.
        return None

    def get_father_type(self, specific_type):
        if not specific_type:
            return None
        # Normalize
        st_clean = specific_type.strip()
        if st_clean in self.father_mapping:
            return self.father_mapping[st_clean]
        if st_clean.lower() in self.father_mapping:
            return self.father_mapping[st_clean.lower()]
        return None

class ImageDataset(Dataset):
    """Load Gallery Images"""
    def __init__(self, sample_ids, data_root):
        self.sample_ids = sample_ids
        self.data_root = Path(data_root)
        self.valid_indices = []
        self._filter_data()

    def _filter_data(self):
        logger.info("Filtering invalid image samples...")
        valid_indices = []
        for idx, sample_id in enumerate(tqdm(self.sample_ids, desc="Checking images")):
            img_path = self.data_root / sample_id / "chart.png"
            if img_path.exists():
                valid_indices.append(idx)
        self.valid_indices = valid_indices
        logger.info(f"Filtered {len(self.sample_ids) - len(self.valid_indices)} invalid images. Remaining: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        sample_id = self.sample_ids[original_idx]

        img_path = self.data_root / sample_id / "chart.png"
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        return {'id': sample_id, 'image': image}

def image_collate_fn(batch, processor):
    images = [item['image'] for item in batch]
    ids = [item['id'] for item in batch]
    image_inputs = processor(images=images, return_tensors="pt", padding=True)["pixel_values"]
    return {'image_inputs': image_inputs, 'ids': ids}

def load_gallery_ids(split_path, split_key='val'):
    """加载所有验证集ID作为Gallery"""
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    if split_key == 'all':
        return split_data.get('val') + split_data.get('train')
    return split_data.get('val', [])

def load_gallery_metadata(metadata_path, hierarchy_path, gallery_ids):
    """
    加载 metadata JSON，提取每个 ID 的 chart type，并映射到 Root Type。
    """
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # Init Mapper
    mapper = ChartTypeMapper(hierarchy_path)

    # 构建 ID -> Info 映射
    logger.info("Indexing metadata...")
    data_map = {}
    for item in data_list:
        fid = item.get('folder_name')
        if fid:
            data_map[fid] = item

    id_to_root_type = {}
    id_to_raw_type = {}
    missing_count = 0

    for sample_id in tqdm(gallery_ids, desc="Parsing metadata"):
        info = data_map.get(sample_id)
        if not info:
            id_to_root_type[sample_id] = None
            id_to_raw_type[sample_id] = None
            if missing_count < 20:
                tqdm.write(f"WARNING: ID {sample_id} not found in metadata file!")
            missing_count += 1
            continue

        # 读取原始 specific type
        raw_type = info.get('chart_type')
        if not raw_type:
            sub_info = info.get('info', {})
            if isinstance(sub_info, dict):
                raw_type = sub_info.get('chart_type')

        # Map to Root Type
        root_type = mapper.get_root_type(raw_type)
        father_type = mapper.get_father_type(raw_type)
        if root_type is None:
            missing_count += 1
        id_to_raw_type[sample_id] = father_type
        id_to_root_type[sample_id] = root_type

    if missing_count > 0:
        logger.warning(f"Failed to map chart type for {missing_count}/{len(gallery_ids)} images. These will have 0 chart type similarity.")

    return id_to_root_type, id_to_raw_type

def load_human_eval_queries(json_path):
    """从人工标注加载Query"""
    logger.info(f"Loading human eval queries from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = []
    for sample_id, info in data.items():
        caption = info.get('correction', '').strip()
        if not caption:
            caption = info.get('original_caption', '').strip()
        if caption:
            queries.append({'id': sample_id, 'caption': caption})
    return queries

# ===========================
# 3. LLM Query Expansion
# ===========================

# Reverting to Positive Multi-Select Prompt
LLM_PROMPT_TEMPLATE = """You are an expert in chart information retrieval. A user has provided a search query for a chart.
Your task is to decompose this single query into 5 distinct aspects.

**CRITICAL RULE**: Be FAITHFUL to the user's query.
- **DO NOT Hallucinate**: Do NOT invent colors, styles, or specific data values that are not mentioned or strongly implied by the user.
- **If Not Mentioned, Leave Empty**: If the user does not describe a specific aspect (e.g., they didn't mention colors), leave the "query" field for that aspect EMPTY (""). Do not guess "blue and gray" or "minimalist".
- **Infer Logic, Not Details**: You can infer structural logic (e.g., "compare" implies multiple bars/lines), but do not infer specific visual details (colors, icons) unless stated.

**Aspect Definitions:**
* **CONTENT**: The topic, data subject, specific metrics, entities, and relationships mentioned.
* **STYLE**: Visual aesthetic, color palette, artistic style (Only if explicitly described, e.g., "dark mode", "blue bars").
* **LAYOUT**: Structural composition and arrangement (Only if explicitly described, e.g., "grid", "horizontal").
* **ILLUSTRATION**: Decorative elements, icons, background graphics (Only if explicitly described).
* **CHART_TYPE**: Identify the specific chart type(s).
  - Allowed: "Bar Chart", "Line Chart", "Area Chart", "Radar Chart", "Pie Chart", "Scatterplot", "Gauge Chart", "Treemap", "Diagram", "Histogram", "Range Chart", "Funnel Chart", "Pyramid Chart".
  - If ambiguous/hybrid, list ALL plausible types (comma-separated).

**Instructions:**
1. **Extraction**: For each aspect, extract the relevant description from the user's query. Rewrite it into a clear phrase. **If the user query does not contain info for an aspect, return an empty string.**
2. **Weight**: Assign a confidence "weight" (0.0 to 1.0).
   - If the aspect is empty, weight MUST be 0.0.
   - If present, weight reflects importance.
3. **Chart Type**:
   - Predict the chart type based on the query.
   - Assign a confidence score (0.0 to 1.0).

User Query: "{user_query}"

Respond with a valid JSON object ONLY:
{{
  "content": {{"query": "...", "weight": 0.x}},
  "style": {{"query": "...", "weight": 0.x}},
  "layout": {{"query": "...", "weight": 0.x}},
  "illustration": {{"query": "...", "weight": 0.x}},
  "chart_type": {{"query": "Type1, Type2", "weight": 0.x}}
}}
"""

def call_llm_api(user_query, api_key, model="Qwen/Qwen3-8B", client=None):
    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
            timeout=20.0
        )

    prompt = LLM_PROMPT_TEMPLATE.format(user_query=user_query)

    for attempt in range(3): # Retry loop
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                timeout=20.0
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON
            if "```" in content:
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)

            data = json.loads(content)

            # Simple validation
            required_keys = ASPECTS + [CHART_TYPE_ASPECT]
            if all(k in data for k in required_keys):
                # Normalize chart type value
                ct_raw = data[CHART_TYPE_ASPECT].get('query', '')
                if ct_raw:
                    # Split comma separated values
                    potential_types = [t.strip() for t in ct_raw.split(',')]
                    valid_types = []

                    for pt in potential_types:
                        if not pt: continue

                        # Exact match check
                        if pt in CHART_TYPE_POOL:
                            valid_types.append(pt)
                            continue

                        # Case insensitive check
                        found = False
                        for pool_t in CHART_TYPE_POOL:
                            if pool_t.lower() == pt.lower():
                                valid_types.append(pool_t)
                                found = True
                                break

                        if not found:
                            logger.warning(f"LLM generated invalid chart type: '{pt}' for query: '{user_query[:50]}...'. It was discarded.")

                    if valid_types:
                        data[CHART_TYPE_ASPECT]['query'] = ", ".join(valid_types)
                    else:
                        data[CHART_TYPE_ASPECT]['query'] = ""
                        data[CHART_TYPE_ASPECT]['weight'] = 0.0

                return data

        except Exception as e:
            logger.warning(f"LLM API Call failed for query '{user_query}' (Attempt {attempt+1}/3): {e}")
            time.sleep(1)

    # Fallback
    logger.error(f"Failed to expand query '{user_query}' after retries. Using fallback.")
    fallback = {k: {"query": "", "weight": 0.0} for k in ASPECTS + [CHART_TYPE_ASPECT]}
    fallback["content"] = {"query": user_query, "weight": 1.0}
    return fallback

def expand_queries_with_llm(queries, cache_file, api_key, max_workers=10):
    expanded_queries = {}

    # Check if cache exists (should probably clean old cache since logic changed back)
    # Using 'human_queries_5types_expanded_multilabel.json'

    if os.path.exists(cache_file):
        logger.info(f"Loading expanded queries from cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            expanded_queries = json.load(f)

    pending_items = []
    for item in queries:
        qid = item['id']
        if qid not in expanded_queries or 'chart_type' not in expanded_queries[qid].get('aspects', {}):
            pending_items.append(item)

    if not pending_items:
        logger.info("All queries already expanded (found in cache).")
        return expanded_queries

    logger.info(f"Expanding {len(pending_items)} pending queries with LLM (Max workers: {max_workers})...")

    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    def process_item(item):
        qid = item['id']
        caption = item['caption']
        expanded = call_llm_api(caption, api_key, client=client)
        return qid, caption, expanded

    # Run concurrency
    updates_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {executor.submit(process_item, item): item['id'] for item in pending_items}

        for future in tqdm(as_completed(future_to_qid), total=len(pending_items), desc="LLM Expansion"):
            try:
                qid, caption, expanded = future.result()
                expanded_queries[qid] = {
                    "original_caption": caption,
                    "aspects": expanded
                }
                updates_count += 1

                if updates_count % 50 == 0:
                     with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Error processing query {future_to_qid[future]}: {e}")

    if updates_count > 0:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(expanded_queries, f, indent=2, ensure_ascii=False)

    return expanded_queries

# ===========================
# 4. Visualization Helpers
# ===========================

def visualize_results(output_dir, query_info, retrieved_items, data_root):
    """
    Visualizes query info and top-5 retrieved images in a single figure.
    Saves the figure to output_dir/visualization/{query_id}/result.png
    """
    qid = query_info['id']
    original_query = query_info['original_query']
    aspects = query_info['aspects']

    # Setup Paths
    vis_dir = Path(output_dir) / "visualization" / qid
    os.makedirs(vis_dir, exist_ok=True)

    # Create Figure: Grid 2 rows x 3 columns (roughly)
    # Top row: Query Info (Text) + GT Image (if available/applicable, here we treat Top-1 as best proxy if we don't have explicit GT path easily accessible,
    # but we can try to find GT image if qid exists in data_root)
    # Actually, qid IS the GT folder name usually.

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 6)

    # --- 1. Query Information Panel (Top Left) ---
    ax_text = fig.add_subplot(gs[0, 0:2])
    ax_text.axis('off')

    # Format Text
    text_content = f"Query ID: {qid}\n\n"
    text_content += f"Original Query:\n{textwrap.fill(original_query, width=40)}\n\n"
    text_content += "LLM Rewrites:\n"

    for asp in ASPECTS:
        w = aspects.get(asp, {}).get('weight', 0.0)
        q = aspects.get(asp, {}).get('query', '')
        if w > 0:
            text_content += f"[{asp.upper()}] (w={w:.2f}):\n{textwrap.fill(q, width=40)}\n"

    ct_info = aspects.get('chart_type', {})
    text_content += f"\n[CHART_TYPE] (w={ct_info.get('weight',0):.2f}): {ct_info.get('query', 'N/A')}"

    ax_text.text(0, 1, text_content, va='top', fontsize=10, family='monospace')

    # --- 2. Ground Truth Image (Top Right) ---
    ax_gt = fig.add_subplot(gs[0, 2:4])
    gt_path = Path(data_root) / qid / "chart.png"
    if gt_path.exists():
        try:
            img = Image.open(gt_path).convert('RGB')
            ax_gt.imshow(img)
            ax_gt.set_title("Ground Truth", color='green', fontweight='bold')
        except:
            ax_gt.text(0.5, 0.5, "Image Load Failed", ha='center')
    else:
        ax_gt.text(0.5, 0.5, "GT Image Not Found", ha='center')
    ax_gt.axis('off')

    # --- 3. Top-5 Retrieved Images (Bottom Row) ---
    for i, item in enumerate(retrieved_items[:5]):
        ax = fig.add_subplot(gs[1, i]) # Bottom row, columns 0-4

        ret_id = item['id']
        score = item['score']
        is_correct = item['is_correct']

        img_path = Path(data_root) / ret_id / "chart.png"

        try:
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Img Not Found", ha='center')
        except:
             ax.text(0.5, 0.5, "Load Error", ha='center')

        title_color = 'green' if is_correct else 'red'
        ax.set_title(f"Rank {i+1}\nID: {ret_id}\nScore: {score:.4f}", color=title_color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(vis_dir / "result_viz.png", dpi=100)
    plt.close(fig)


# ===========================
# 5. Evaluation Logic
# ===========================

def compute_metrics(final_sim, query_ids_list, gallery_ids_list, expanded_query_map, raw_queries, mode_name, aspects_list, visualize=False, data_root=None, output_dir=None):
    """
    Common metric computation function
    """
    logger.info(f"Calculating Metrics for {mode_name}...")
    n_qry = len(query_ids_list)
    t2i_scores, t2i_indices = torch.topk(final_sim, k=10, dim=1)

    top1 = 0
    top5 = 0
    mrr_sum = 0.0

    aspect_stats = {
        k: {'total': 0, 'top1': 0, 'top5': 0, 'mrr_sum': 0.0}
        for k in aspects_list if k in ASPECTS
    }

    retrieval_results = []

    for i in range(n_qry):
        q_id = query_ids_list[i]

        # Determine Dominant Aspect based on LLM weights (only dense ones)
        current_aspects = expanded_query_map.get(q_id, {}).get('aspects', {})
        max_weight = -1.0
        dominant_aspect = 'content'

        for asp in ASPECTS:
            aspect_data = current_aspects.get(asp, {})
            if not isinstance(aspect_data, dict):
                w = 0.0
            else:
                w = aspect_data.get('weight')
                if w is None:
                    w = 0.0

            if w > max_weight:
                max_weight = w
                dominant_aspect = asp

        if max_weight <= 0:
            dominant_aspect = 'content'

        if dominant_aspect in aspect_stats:
            aspect_stats[dominant_aspect]['total'] += 1

        is_hit_top1 = False
        is_hit_top5 = False
        rank_position = -1

        top_k_items = []

        for rank in range(5):
            idx = t2i_indices[i, rank].item()
            score = t2i_scores[i, rank].item()
            retrieved_id = gallery_ids_list[idx]

            is_correct = (retrieved_id == q_id)
            if is_correct:
                is_hit_top5 = True
                if rank == 0:
                    is_hit_top1 = True

            top_k_items.append({
                "rank": rank + 1,
                "id": retrieved_id,
                "score": score,
                "is_correct": is_correct
            })

        for rank in range(10):
            idx = t2i_indices[i, rank].item()
            if gallery_ids_list[idx] == q_id:
                rank_position = rank + 1
                mrr_sum += 1.0 / rank_position
                if dominant_aspect in aspect_stats:
                    aspect_stats[dominant_aspect]['mrr_sum'] += 1.0 / rank_position
                break

        if is_hit_top1:
            top1 += 1
            if dominant_aspect in aspect_stats:
                aspect_stats[dominant_aspect]['top1'] += 1
        if is_hit_top5:
            top5 += 1
            if dominant_aspect in aspect_stats:
                aspect_stats[dominant_aspect]['top5'] += 1

        retrieval_results.append({
            "query_id": q_id,
            "original_query": raw_queries[i]['caption'],
            "dominant_aspect": dominant_aspect,
            "retrieved": top_k_items,
            "rank": rank_position
        })

        # Visualization
        if visualize and data_root and output_dir:
            # We construct a query_info object for visualization
            q_info = {
                "id": q_id,
                "original_query": raw_queries[i]['caption'],
                "aspects": current_aspects
            }
            visualize_results(output_dir, q_info, top_k_items, data_root)

    acc_top1 = top1 / n_qry if n_qry > 0 else 0
    acc_top5 = top5 / n_qry if n_qry > 0 else 0
    mrr = mrr_sum / n_qry if n_qry > 0 else 0

    result_str = (
        f"[{mode_name}] Overall Metrics:\n"
        f"  Top-1 Accuracy: {acc_top1*100:.2f}%\n"
        f"  Top-5 Accuracy: {acc_top5*100:.2f}%\n"
        f"  MRR@10        : {mrr:.4f}\n"
        f"Breakdown by Dominant Aspect:\n"
    )

    for aspect in aspect_stats:
        stats = aspect_stats[aspect]
        total = stats['total']
        if total > 0:
            asp_top1 = stats['top1'] / total * 100
            asp_top5 = stats['top5'] / total * 100
            asp_mrr = stats['mrr_sum'] / total
            result_str += f"  [{aspect.upper()}] (N={total}): Top-1={asp_top1:.2f}%, Top-5={asp_top5:.2f}%, MRR={asp_mrr:.4f}\n"
        else:
            pass

    return result_str, retrieval_results


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Model ---
    model = MultiHeadBGEVL(args.base_model)
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    processor = model.processor

    # --- 2. Prepare Gallery (All Validation Images) ---
    gallery_ids = load_gallery_ids(args.split_file)
    if args.limit_val > 0:
        gallery_ids = gallery_ids[:args.limit_val]

    gallery_dataset = ImageDataset(gallery_ids, args.data_root)
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=lambda b: image_collate_fn(b, processor)
    )

    # --- 2.5 Load Gallery Metadata (Chart Types) with Hierarchy Mapping ---
    gallery_type_map = load_gallery_metadata(args.metadata_file, args.hierarchy_file, gallery_ids)

    # --- 3. Extract Gallery Features ---
    gallery_embeds = {k: [] for k in ASPECTS}
    gallery_ids_list = []

    logger.info("Extracting Gallery Features...")
    with torch.no_grad():
        for batch in tqdm(gallery_loader):
            image_inputs = batch['image_inputs'].to(device)
            batch_ids = batch['ids']
            gallery_ids_list.extend(batch_ids)

            feats = model.forward_image_features(image_inputs)
            for aspect in ASPECTS:
                gallery_embeds[aspect].append(feats[aspect].cpu())

    for aspect in ASPECTS:
        gallery_embeds[aspect] = torch.cat(gallery_embeds[aspect], dim=0) # (N_gal, D)

    # --- 4. Prepare & Expand Queries ---
    raw_queries = load_human_eval_queries(args.human_file)
    if args.limit_val > 0:
        raw_queries = raw_queries[:args.limit_val]

    # Expand using LLM - Positive Multi-Label
    cache_path = Path(args.output_dir) / "human_queries_5types_expanded_multilabel_v3.json"
    os.makedirs(args.output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        api_key = "sk-nfbkmewiqeilanriqoxgutbhvzbfbtgxtnmevmkaetyxfdon"

    expanded_query_map = expand_queries_with_llm(raw_queries, cache_path, api_key, max_workers=args.max_workers)

    # --- 5. Extract Query Features ---
    query_ids_list = [q['id'] for q in raw_queries]

    aspect_texts_map = {k: [] for k in ASPECTS}
    aspect_weights_map = {k: [] for k in ASPECTS}
    chart_type_info_list = []

    for q_item in raw_queries:
        qid = q_item['id']
        info = expanded_query_map.get(qid, {}).get('aspects', {})

        # Dense aspects
        for aspect in ASPECTS:
            aspect_data = info.get(aspect, {"query": "", "weight": 0.0})
            text = aspect_data.get("query", "")
            weight = aspect_data.get("weight", 0.0)

            if text.strip():
                text_input = PREFIXES[aspect] + text
            else:
                text_input = ""

            aspect_texts_map[aspect].append(text_input)
            aspect_weights_map[aspect].append(weight)

        # Sparse aspect: Chart Type (Positive Multi-Label)
        ct_data = info.get(CHART_TYPE_ASPECT, {"query": "", "weight": 0.0})
        raw_types = ct_data.get('query', '')
        weight = ct_data.get('weight', 0.0)

        if raw_types:
            type_list = [t.strip() for t in raw_types.split(',')]
        else:
            type_list = []

        chart_type_info_list.append({
            'types': type_list,
            'confidence': weight
        })

    # Compute Embeddings
    query_embeds = {}

    logger.info("Extracting Query Features for dense aspects...")
    for aspect in ASPECTS:
        texts = aspect_texts_map[aspect]
        all_feats = []
        batch_size = args.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            processed_texts = [t if t.strip() else PREFIXES[aspect]+"chart" for t in batch_texts]

            inputs = processor(text=processed_texts, return_tensors="pt", padding=True, max_length=77, truncation=True)
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                feat = model.forward_text_features(input_ids, aspect)
                all_feats.append(feat.cpu())

        query_embeds[aspect] = torch.cat(all_feats, dim=0) # (N_qry, D)

    # --- 6. Compute Similarity Matrices ---
    n_qry = len(query_ids_list)
    n_gal = len(gallery_ids_list)

    # 6.1 Dense Aspects Similarity (Per aspect)
    dense_sim_map = {}
    for aspect in ASPECTS:
        dense_sim_map[aspect] = torch.matmul(query_embeds[aspect], gallery_embeds[aspect].T)

    # 6.2 Chart Type Similarity (Multi-Label Match with Soft Similarity)
    logger.info("Computing Chart Type Similarity (Positive Multi-Label + Soft Sim)...")
    type_to_id = {t: i+1 for i, t in enumerate(CHART_TYPE_POOL)}
    num_types = len(CHART_TYPE_POOL)

    # Build Similarity Matrix: (Num_Types+1, Num_Types+1)
    # Default diagonal = 1.0
    sim_matrix = torch.eye(num_types + 1, device='cpu')

    # Fill in expert defined similarities
    for pair, score in CHART_TYPE_SIMILARITY.items():
        types = list(pair)
        if len(types) == 2:
            t1, t2 = types[0], types[1]
            if t1 in type_to_id and t2 in type_to_id:
                id1, id2 = type_to_id[t1], type_to_id[t2]
                sim_matrix[id1, id2] = score
                sim_matrix[id2, id1] = score

    # Zero out Unknown (Index 0) rows/cols to be safe (though identity handles it, we don't want 0 matching 0)
    sim_matrix[0, :] = 0.0
    sim_matrix[:, 0] = 0.0

    # Gallery One-Hot
    gal_onehot = torch.zeros((n_gal, num_types + 1), device='cpu')
    for idx, gid in enumerate(gallery_ids_list):
        t = gallery_type_map.get(gid)
        tid = type_to_id.get(t, 0) # 0 if None
        gal_onehot[idx, tid] = 1.0

    # Query Multi-Hot
    qry_multihot = torch.zeros((n_qry, num_types + 1), device='cpu')
    qry_confidences = []

    for idx, info in enumerate(chart_type_info_list):
        types = info['types']
        conf = info['confidence']
        qry_confidences.append(conf)

        if types:
            for t in types:
                tid = type_to_id.get(t, 0)
                if tid > 0:
                    qry_multihot[idx, tid] = 1.0

    # Calculation: Q @ SimMatrix @ G.T
    # First, expand query types by similarity
    qry_expanded = torch.matmul(qry_multihot, sim_matrix) # (N_qry, T)

    # Then match with gallery
    # This gives the Soft Similarity Score
    chart_type_sim = torch.matmul(qry_expanded, gal_onehot.T) # (N_qry, N_gal)

    # --- DEBUG STATISTICS ---
    logger.info("--- Chart Type Statistics ---")
    valid_gal_types = (gal_onehot.sum(dim=1) > 0).sum().item()
    valid_qry_types = (qry_multihot.sum(dim=1) > 0).sum().item()

    logger.info(f"Gallery Images with Mapped Root Type: {valid_gal_types}/{n_gal}")
    logger.info(f"Queries with Mapped Root Type: {valid_qry_types}/{n_qry}")

    # --- 7. Evaluation ---

    # 7.1 4-Types (Baseline)
    sim_4types = torch.zeros((n_qry, n_gal))
    norm_factors = torch.zeros(n_qry)
    for aspect in ASPECTS:
        norm_factors += torch.tensor(aspect_weights_map[aspect])
    norm_factors[norm_factors == 0] = 1.0

    for aspect in ASPECTS:
        w = torch.tensor(aspect_weights_map[aspect]).unsqueeze(1)
        w_norm = w / norm_factors.unsqueeze(1)
        sim_4types += w_norm * dense_sim_map[aspect]

    str_4types, res_4types = compute_metrics(
        sim_4types, query_ids_list, gallery_ids_list, expanded_query_map, raw_queries,
        "4-Types (Baseline)", ASPECTS,
        visualize=False # Only visualize for the main 5-types model if requested
    )

    # 7.2 5-Types (Positive Match)
    sim_5types = torch.zeros((n_qry, n_gal))
    total_weights = norm_factors.clone() + torch.tensor(qry_confidences)
    total_weights[total_weights == 0] = 1.0

    for aspect in ASPECTS:
        w = torch.tensor(aspect_weights_map[aspect]).unsqueeze(1)
        w_norm = w / total_weights.unsqueeze(1)
        sim_5types += w_norm * dense_sim_map[aspect]

    w_ct = torch.tensor(qry_confidences).unsqueeze(1)
    w_ct_norm = w_ct / total_weights.unsqueeze(1)

    sim_5types += w_ct_norm * chart_type_sim

    # Visualize only for the final 5-types model
    str_5types, res_5types = compute_metrics(
        sim_5types, query_ids_list, gallery_ids_list, expanded_query_map, raw_queries,
        "5-Types (Multi-Label Positive Match)", ASPECTS + [CHART_TYPE_ASPECT],
        visualize=args.visualize, data_root=args.data_root, output_dir=args.output_dir
    )

    # --- 7.3 Metrics: Alignment Rate & LLM Recall ---

    # 1. LLM Chart Type Recall
    logger.info("Calculating LLM Chart Type Recall (GT Type vs LLM Prediction)...")
    llm_recall_count = 0
    valid_gt_count = 0

    # DEBUG: Track recall failures
    recall_failures = []

    for i in range(n_qry):
        q_id = query_ids_list[i]
        q_text = raw_queries[i]['caption']
        llm_types_str = chart_type_info_list[i]['types']

        gt_type = gallery_type_map.get(q_id)
        valid_pred_indices = torch.nonzero(qry_multihot[i]).flatten().tolist()

        if gt_type:
            valid_gt_count += 1
            gt_type_id = type_to_id.get(gt_type, 0)

            if valid_pred_indices and gt_type_id in valid_pred_indices:
                llm_recall_count += 1
            else:
                if len(recall_failures) < 10:
                    recall_failures.append({
                        "query": q_text[:100],
                        "llm_pred": llm_types_str,
                        "gt_type": gt_type
                    })

    llm_recall = llm_recall_count / valid_gt_count if valid_gt_count > 0 else 0
    logger.info(f"LLM Chart Type Recall: {llm_recall*100:.2f}% (on {valid_gt_count} queries with valid GT)")

    if recall_failures:
        logger.info("\n--- LLM Recall Failure Examples (GT not in Prediction) ---")
        for ex in recall_failures:
            logger.info(f"Query: {ex['query']}...")
            logger.info(f"  LLM Predicted: {ex['llm_pred']}")
            logger.info(f"  Ground Truth: {ex['gt_type']}")
            logger.info("---")

    # 2. Alignment Rate @ 10
    logger.info("Calculating Alignment Rate (Top-10 Dense Results vs LLM Chart Type)...")

    _, top10_indices_4types = torch.topk(sim_4types, k=10, dim=1)

    alignment_count = 0
    total_slots = n_qry * 10

    for i in range(n_qry):
        valid_type_indices = torch.nonzero(qry_multihot[i]).flatten().tolist()

        if not valid_type_indices:
            continue

        for rank in range(10):
            idx = top10_indices_4types[i, rank].item()
            gal_type_idx = torch.argmax(gal_onehot[idx]).item()

            if gal_type_idx in valid_type_indices:
                alignment_count += 1

    alignment_rate = alignment_count / total_slots if total_slots > 0 else 0
    logger.info(f"Alignment Rate @ 10: {alignment_rate*100:.2f}%")

    # --- Output Results ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_output = (
        f"\n{'='*60}\n"
        f"Time: {timestamp}\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Comparison: 4-Types vs 5-Types (With Hierarchy Mapping)\n"
        f"Queries: {n_qry}, Gallery: {n_gal}\n"
        f"LLM Chart Type Recall (GT in Prediction): {llm_recall*100:.2f}%\n"
        f"Alignment Rate @ 10 (Dense Top-10 matches LLM Type): {alignment_rate*100:.2f}%\n"
        f"{'-'*60}\n"
        f"{str_4types}\n"
        f"{'-'*60}\n"
        f"{str_5types}\n"
        f"{'='*60}\n"
    )

    print(final_output)

    if args.output_dir:
        out_path = Path(args.output_dir)
        with open(out_path / "eval_results_comparison_multilabel.txt", "a") as f:
            f.write(final_output)

        json_path_4 = out_path / f"retrieval_details_4types_{timestamp.replace(' ', '_')}.json"
        with open(json_path_4, 'w', encoding='utf-8') as f:
            json.dump(res_4types, f, indent=2, ensure_ascii=False)

        json_path_5 = out_path / f"retrieval_details_5types_multilabel_{timestamp.replace(' ', '_')}.json"
        with open(json_path_5, 'w', encoding='utf-8') as f:
            json.dump(res_5types, f, indent=2, ensure_ascii=False)

        logger.info(f"Details saved to output directory")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="BAAI/BGE-VL-base")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pt")
    parser.add_argument("--data_root", type=str, default="/mnt/share/public/converted/converted")
    parser.add_argument("--human_file", type=str, default="/mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/retrieval_training/data/human_eval_results.json")
    parser.add_argument("--split_file", type=str, default="/mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/retrieval_training/data/train_filtered_thresh_0.92.json")
    parser.add_argument("--metadata_file", type=str, default="/mnt/share/xujing/Chart_Retrieval/VizRetrieval/data/samples_info_200k_new.json")
    parser.add_argument("--hierarchy_file", type=str, default="/mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/data_processing/chart_types_hierarchy.json")
    parser.add_argument("--limit_val", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="/mnt/share/xujing/Chart_Retrieval/VizRetrieval/ChartRetrieval/retrieval_training/output_5types/human_eval")
    parser.add_argument("--api_key", type=str, default=None, help="SiliconFlow API Key")
    parser.add_argument("--max_workers", type=int, default=5, help="Max workers for LLM expansion")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of retrieval results")

    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
