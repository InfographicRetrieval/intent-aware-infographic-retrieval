"""
Baseline Retrieval Model - BGE-VL
直接使用 BGE-VL-base 的基础模型，不加载微调权重
与 RetrievalV3 接口兼容
"""

import os
import sys
import json
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from PIL import Image

# ================= 配置路径 =================
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent

DATA_ROOT = os.environ.get("CHARTRETRIEVAL_DATA_ROOT", "/mnt/share/public/converted/converted")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将父目录加入 path 以便导入工具函数
PARENT_DIR = str(REPO_ROOT / "retrieval_training")
sys.path.append(PARENT_DIR)

from evaluate_5types_human import load_gallery_ids

# ================= 资源配置 =================
BASE_MODEL_NAME = "BAAI/BGE-VL-base"
EMBEDDINGS_CACHE_DIR = os.environ.get("RETRIEVAL_EMBEDDINGS_CACHE_DIR", str(REPO_ROOT / "retrieval_training" / "data" / "embeddings_cache"))
SPLIT_FILE = os.environ.get("RETRIEVAL_SPLIT_FILE", str(REPO_ROOT / "retrieval_training" / "data" / "train_filtered_thresh_0.92.json"))
SET_INFO = os.environ.get("CHART_METADATA_FILE", str(WORKSPACE_ROOT / "data" / "samples_info_200k_new.json"))
SET_NAME = 'all'


class BaselineImageDataset(Dataset):
    """Baseline 用的图像数据集"""
    def __init__(self, sample_ids, data_root):
        self.sample_ids = sample_ids
        self.data_root = Path(data_root)
        self.valid_indices = []
        self._filter_data()
        
    def _filter_data(self):
        for idx, sample_id in enumerate(self.sample_ids):
            img_path = self.data_root / sample_id / "chart.png"
            if img_path.exists():
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        sample_id = self.sample_ids[original_idx]
        img_path = self.data_root / sample_id / "chart.png"
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='white')
        return {'id': sample_id, 'image': image}


class RetrievalBaseline:
    """
    Baseline 检索器 - BGE-VL
    - 使用 BGE-VL-base 基础模型（不加载微调权重）
    - 直接使用文本和图像特征，不加 aspect 前缀
    - 与 RetrievalV3 接口兼容
    """
    
    def __init__(self):
        # 确保缓存目录存在
        os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
        
        # 1. 加载 BGE-VL 基础模型
        print("Loading Baseline BGE-VL model...")
        self.model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        self.model.eval()
        print("    BGE-VL model loaded (no fine-tuned weights).")
        
        # 2. 加载 Gallery IDs
        print("Loading gallery IDs...")
        if os.path.exists(SPLIT_FILE):
            self.all_gallery_ids = load_gallery_ids(SPLIT_FILE, SET_NAME)
            self.all_gallery_ids = sorted(list(set(self.all_gallery_ids)))
            print(f"    Loaded {len(self.all_gallery_ids)} gallery IDs")
        else:
            raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")
        
        # 3. 加载 Gallery 元数据（用于获取 chart type）
        # self.gallery_type_map = self._load_gallery_type_map()
        
        # 4. 加载/提取 Gallery Features
        print("Loading/Extracting gallery features...")
        self.gallery_features = self._get_gallery_features()
        print("    Gallery features ready!")

    # def _load_gallery_type_map(self):
    #     """加载 gallery 的 chart type 映射"""
    #     type_map = {}
    #     if os.path.exists(SET_INFO):
    #         with open(SET_INFO, 'r', encoding='utf-8') as f:
    #             data = json.load(f)
    #         for sample_id in self.all_gallery_ids:
    #             info = data.get(sample_id, {})
    #             chart_type = info.get('chart_type', 'unknown')
    #             type_map[sample_id] = chart_type
    #     return type_map
    
    def _get_gallery_features(self):
        """
        预计算所有候选图片的特征，并持久化到磁盘
        """
        num_samples = len(self.all_gallery_ids)
        cache_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"gallery_embeds_Baseline_BGE-VL_{SET_NAME}_{num_samples}.pt")
        
        # 检查缓存
        if os.path.exists(cache_path):
            print(f"    Loading cached embeddings...")
            cached_data = torch.load(cache_path, map_location='cpu')
            if len(cached_data["ids"]) == num_samples:
                print("    Cache loaded!")
                return cached_data
            else:
                print(f"    Cache mismatch. Re-extracting...")
        
        # 提取图片特征
        print(f"    Extracting features (this may take a while)...")
        dataset = BaselineImageDataset(self.all_gallery_ids, DATA_ROOT)
        
        def collate_fn(batch):
            return {'images': [item['image'] for item in batch], 'ids': [item['id'] for item in batch]}
        
        loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, num_workers=4)
        
        embeds = []
        ids_list = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="    Image Encoding"):
                pixel_values = self.processor(images=batch['images'], return_tensors="pt")["pixel_values"].to(DEVICE)
                vision_outputs = self.model.vision_model(pixel_values)
                # BGE-VL 使用 pooler_output (index 1) 和 visual_projection
                feats = self.model.visual_projection(vision_outputs[1])
                # 归一化
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embeds.append(feats.cpu())
                ids_list.extend(batch['ids'])
        
        embeds = torch.cat(embeds, dim=0)
        
        gallery_features = {
            "embeds": embeds,
            "ids": ids_list,
        }
        
        torch.save(gallery_features, cache_path)
        print(f"    Saved to {cache_path}")
        
        return gallery_features
    
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """编码文本，返回归一化的特征"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.processor(
            text=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(DEVICE)
        
        with torch.no_grad():
            text_outputs = self.model.text_model(inputs["input_ids"])
            feats = self.model.text_projection(text_outputs[1])  # pooler_output
            # 归一化
            feats = feats / feats.norm(dim=-1, keepdim=True)
        
        return feats.cpu()
    
    def retrieve(self, query: Any, k: int = 10) -> List[Dict]:
        """
        执行 Baseline 检索
        
        Args:
            query: 可以是字符串或 5-aspect dict
            k: 返回结果数量
            
        Returns:
            Top-k 结果列表，格式与 RetrievalV3.retrieve() 兼容
        """
        # 1. 处理输入 query，提取文本
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, dict):
            # 如果是 5-aspect dict，尝试提取 content，否则拼接所有非空 query
            content_data = query.get("content", {})
            if isinstance(content_data, dict):
                query_text = content_data.get("query", "")
            else:
                query_text = str(content_data)
            
            # 如果没有 content，尝试用其他 aspects 拼接
            if not query_text:
                parts = []
                for aspect in ["content", "style", "layout", "illustration", "chart_type"]:
                    asp_data = query.get(aspect, {})
                    if isinstance(asp_data, dict):
                        text = asp_data.get("query", "")
                        if text:
                            parts.append(text)
                query_text = " ".join(parts)
        else:
            query_text = str(query)
        
        if not query_text.strip():
            query_text = "chart"
        
        # 2. 编码 query
        query_feat = self._encode_text([query_text])
        
        # 3. 计算相似度
        gal_ids = self.gallery_features["ids"]
        gal_embeds = self.gallery_features["embeds"]
        
        # 余弦相似度（已经归一化，直接点积）
        sim = torch.matmul(query_feat, gal_embeds.T).squeeze(0)
        
        # 4. 取 Top-K
        topk_scores, topk_indices = torch.topk(sim, k=min(k, len(gal_ids)))
        
        # 5. 构造结果
        results = []
        for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), 1):
            gallery_id = gal_ids[idx.item()]
            result = {
                'rank': rank,
                'similarity_score': float(score),
                'folder_name': gallery_id,
                'chart_path': f"{gallery_id}/chart.png"
            }
            results.append(result)
        
        return results
    
    def search(self, query: Any, top_k: int = 10, **kwargs) -> Dict:
        """
        兼容 RetrievalV3.search() 的接口
        
        Args:
            query: 字符串或 5-aspect dict
            top_k: 返回结果数量
            **kwargs: 兼容其他参数
            
        Returns:
            Dict 格式与 RetrievalV3.search() 兼容
        """
        import time
        start_time = time.time()
        
        results = self.retrieve(query, k=top_k)
        
        search_time = time.time() - start_time
        
        return {
            'query_info': {
                'type': 'baseline_bge_vl',
                'timestamp': start_time
            },
            'results': results,
            'search_stats': {
                'total_found': len(results),
                'search_time': round(search_time, 3),
                'filters_applied': {}
            }
        }


# ================= 便捷函数 =================

_retrieval_baseline_instance = None

def get_retrieval_baseline() -> RetrievalBaseline:
    """获取 Baseline 检索器单例"""
    global _retrieval_baseline_instance
    if _retrieval_baseline_instance is None:
        _retrieval_baseline_instance = RetrievalBaseline()
    return _retrieval_baseline_instance


def baseline_search(query: Any, top_k: int = 20) -> List[Dict]:
    """
    便捷的搜索函数
    
    Args:
        query: 字符串或 5-aspect dict
        top_k: 返回结果数量
        
    Returns:
        Top-k 结果列表
    """
    retriever = get_retrieval_baseline()
    result = retriever.search(query, top_k=top_k)
    return result['results']


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RetrievalBaseline (BGE-VL)")
    print("=" * 60)
    
    retrieval = RetrievalBaseline()
    
    # 测试 1: 字符串 query
    print("\n[Test 1] String Query:")
    query_str = "sales data comparison across quarters"
    print(f"Query: {query_str}")
    results = retrieval.retrieve(query_str, k=5)
    for r in results:
        print(f"  {r['rank']}. {r['chart_path']} (score: {r['similarity_score']:.4f})")
    
    # 测试 2: 5-aspect dict query
    print("\n[Test 2] 5-Aspect Dict Query:")
    query_dict = {
        "content": {"query": "COVID-19 daily mortality rate", "weight": 0.8},
        "chart_type": {"query": "Line Chart", "weight": 0.5}
    }
    results = retrieval.retrieve(query_dict, k=5)
    for r in results:
        print(f"  {r['rank']}. {r['chart_path']} (type: {r['chart_type']}, score: {r['similarity_score']:.4f})")
