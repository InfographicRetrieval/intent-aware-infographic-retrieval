import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ================= 配置路径 =================
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DATA_ROOT = os.environ.get("CHARTRETRIEVAL_DATA_ROOT", "/mnt/share/public/converted/converted")
DEVICE = os.environ.get("CHARTRETRIEVAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# 将父目录加入 path 以便导入算法脚本
PARENT_DIR = str(REPO_ROOT / "retrieval_training")
sys.path.append(PARENT_DIR)

from evaluate_5types_human import (
    MultiHeadBGEVL, load_gallery_ids, ASPECTS, PREFIXES, 
    CHART_TYPE_ASPECT, CHART_TYPE_POOL, CHART_TYPE_SIMILARITY, 
    load_gallery_metadata, ImageDataset, image_collate_fn
)

# ================= 资源配置 =================
BASE_MODEL_NAME = "BAAI/BGE-VL-base"
OUR_MODEL_CKPT = os.environ.get("RETRIEVAL_CKPT", str(REPO_ROOT / "retrieval_training" / "output_4types" / "best_model.pt"))
EMBEDDINGS_CACHE_DIR = os.environ.get("RETRIEVAL_EMBEDDINGS_CACHE_DIR", str(REPO_ROOT / "retrieval_training" / "data" / "embeddings_cache"))
SPLIT_FILE = os.environ.get("RETRIEVAL_SPLIT_FILE", str(REPO_ROOT / "retrieval_training" / "data" / "train_filtered_thresh_0.92.json"))
HIERARCHY_FILE = os.environ.get("CHART_TYPES_HIERARCHY_FILE", str(REPO_ROOT / "data_processing" / "chart_types_hierarchy.json"))
SET_NAME = 'all'
SET_INFO = os.environ.get("CHART_METADATA_FILE", str(WORKSPACE_ROOT / "data" / "samples_info_200k_new.json"))


class RetrievalV3:
    def __init__(self):
        # 确保缓存目录存在
        os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
        
        # 1. 加载 Ours 模型 (带 Checkpoint)
        print("Loading model...")
        self.model = MultiHeadBGEVL(BASE_MODEL_NAME)
        if os.path.exists(OUR_MODEL_CKPT):
            self.model.load_state_dict(torch.load(OUR_MODEL_CKPT, map_location='cpu'))
            print(f"    Loaded fine-tuned weights from {OUR_MODEL_CKPT}")
        self.model.to(DEVICE)
        self.model.eval()
        
        # 2. 加载 Gallery IDs
        print("Loading gallery IDs...")
        if os.path.exists(SPLIT_FILE):
            self.all_gallery_ids = load_gallery_ids(SPLIT_FILE, SET_NAME)
            self.all_gallery_ids = sorted(list(set(self.all_gallery_ids)))
            print(f"    Loaded {len(self.all_gallery_ids)} gallery IDs from {SPLIT_FILE}")
        else:
            raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")
        
        # 3. 加载/提取 Gallery Features
        print("Loading/Extracting gallery features...")
        self.gallery_features, self.gallery_type_map, self.gallery_raw_type_map = self._get_gallery_features()
        print("    Gallery features ready!")

    def _get_gallery_features(self):
        """
        预计算所有候选图片的特征，并持久化到磁盘
        返回: (gallery_features_dict, gallery_type_map)
        """
        num_samples = len(self.all_gallery_ids)
        cache_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"gallery_embeds_ours_{SET_NAME}_{num_samples}.pt")
        
        # 1. 准备 Gallery Metadata (chart type mapping)
        print("    Loading gallery metadata...")
        gallery_type_map, gallery_raw_type_map = load_gallery_metadata(SET_INFO, HIERARCHY_FILE, self.all_gallery_ids)
        
        # 2. 准备 Chart Type One-Hot 编码
        type_to_id = {t: i+1 for i, t in enumerate(CHART_TYPE_POOL)}
        num_types = len(CHART_TYPE_POOL)
        gal_onehot = torch.zeros((num_samples, num_types + 1))
        for idx, gid in enumerate(self.all_gallery_ids):
            gal_onehot[idx, type_to_id.get(gallery_type_map.get(gid), 0)] = 1.0
        
        # 3. 检查缓存
        if os.path.exists(cache_path):
            print(f"    Loading cached embeddings from {cache_path}...")
            cached_data = torch.load(cache_path, map_location='cpu')
            # 校验 ID 数量是否一致
            if len(cached_data["ids"]) == num_samples:
                print("    Cache loaded successfully!")
                return cached_data, gallery_type_map, gallery_raw_type_map
            else:
                print(f"    Cache size mismatch ({len(cached_data['ids'])} vs {num_samples}). Re-extracting...")
        
        # 4. 提取图片特征
        print(f"    Extracting gallery features (this may take a while)...")
        processor = self.model.processor
        gallery_dataset = ImageDataset(self.all_gallery_ids, DATA_ROOT)
        gallery_loader = DataLoader(
            gallery_dataset, batch_size=128, shuffle=False, 
            num_workers=4, collate_fn=lambda b: image_collate_fn(b, processor)
        )
        
        embeds = {k: [] for k in ASPECTS}
        ids_list = []
        
        with torch.no_grad():
            for batch in tqdm(gallery_loader, desc="    Gallery Features"):
                image_inputs = batch['image_inputs'].to(DEVICE)
                batch_ids = batch['ids']
                ids_list.extend(batch_ids)
                
                feats = self.model.forward_image_features(image_inputs)
                for aspect in ASPECTS:
                    embeds[aspect].append(feats[aspect].cpu())
        
        for aspect in ASPECTS:
            embeds[aspect] = torch.cat(embeds[aspect], dim=0)
        
        # 5. 处理缺失图片的情况（调整 one-hot）
        if len(ids_list) != num_samples:
            print(f"    Warning: {num_samples - len(ids_list)} images were missing. Adjusting one-hot...")
            current_onehot = torch.zeros((len(ids_list), num_types + 1))
            id_to_idx = {gid: i for i, gid in enumerate(self.all_gallery_ids)}
            for i, gid in enumerate(ids_list):
                current_onehot[i] = gal_onehot[id_to_idx[gid]]
        else:
            current_onehot = gal_onehot
        
        # 6. 组装并保存
        gallery_features = {
            "embeds": embeds,
            "ids": ids_list,
            "onehot": current_onehot
        }
        
        torch.save(gallery_features, cache_path)
        print(f"    Saved embeddings to {cache_path}")
        
        return gallery_features, gallery_type_map, gallery_raw_type_map

    def retrieve(self, query: Dict, k: int = 10) -> List[Dict]:
        """
        执行检索
        
        Args:
            query: 已经解析的 dict，包含 5 个 aspects（content, style, layout, illustration, chart_type）
                   每个 aspect 应该有 {"query": str, "weight": float} 结构
            k: 返回结果数量
            
        Returns:
            Top-k 结果列表，每个元素是包含 chart_path, chart_type, similarity_score 等信息的 dict
            格式与原来的 MultimodalRetriever.search() 兼容
        """
        processor = self.model.processor
        aspects_info = query
        
        # 1. 提取 Query Features
        query_embeds = {}
        aspect_weights = []
        
        for aspect in ASPECTS:
            asp_data = aspects_info.get(aspect, {"query": "", "weight": 0.0})
            text = asp_data.get("query", "")
            weight = asp_data.get("weight", 0.0)
            
            text_input = PREFIXES[aspect] + text if text.strip() else PREFIXES[aspect] + "chart"
            inputs = processor(text=[text_input], return_tensors="pt", padding=True, max_length=77, truncation=True)
            
            with torch.no_grad():
                feat = self.model.forward_text_features(inputs["input_ids"].to(DEVICE), aspect)
                query_embeds[aspect] = feat.cpu()
            aspect_weights.append(weight)
        
        # 2. Chart Type Similarity
        ct_data = aspects_info.get(CHART_TYPE_ASPECT, {"query": "", "weight": 0.0})
        query_types = [t.strip() for t in ct_data.get('query', '').split(',')] if ct_data.get('query') else []
        ct_weight = ct_data.get('weight', 0.0)
        
        # 准备相似度矩阵
        type_to_id = {t: i+1 for i, t in enumerate(CHART_TYPE_POOL)}
        num_types = len(CHART_TYPE_POOL)
        sim_matrix = torch.eye(num_types + 1)
        for pair, score in CHART_TYPE_SIMILARITY.items():
            t_list = list(pair)
            if len(t_list) == 2:
                t1, t2 = t_list[0], t_list[1]
                if t1 in type_to_id and t2 in type_to_id:
                    id1, id2 = type_to_id[t1], type_to_id[t2]
                    sim_matrix[id1, id2] = sim_matrix[id2, id1] = score
        sim_matrix[0, :] = 0.0
        sim_matrix[:, 0] = 0.0
        
        # Gallery 数据
        gal_ids = self.gallery_features["ids"]
        gal_embeds = self.gallery_features["embeds"]
        gal_onehot = self.gallery_features["onehot"]
        
        # Query Multi-Hot
        qry_multihot = torch.zeros((1, num_types + 1))
        for t in query_types:
            if t in type_to_id:
                qry_multihot[0, type_to_id[t]] = 1.0
        
        # 计算 Chart Type 相似度
        if gal_onehot is not None and query_types:
            chart_type_sim = torch.matmul(torch.matmul(qry_multihot, sim_matrix), gal_onehot.T).squeeze(0)
        else:
            chart_type_sim = torch.zeros(len(gal_ids))
        
        # 3. 计算最终的加权相似度
        total_weight = sum(aspect_weights) + ct_weight
        if total_weight == 0:
            total_weight = 1.0
        
        final_sim = torch.zeros(len(gal_ids))
        
        for i, aspect in enumerate(ASPECTS):
            w = aspect_weights[i]
            sim = torch.matmul(query_embeds[aspect], gal_embeds[aspect].T).squeeze(0)
            final_sim += (w / total_weight) * sim
        
        # 加上 chart type 相似度
        if ct_weight > 0:
            final_sim += (ct_weight / total_weight) * chart_type_sim
        
        # 4. 取 Top-K 并构造结果
        topk_scores, topk_indices = torch.topk(final_sim, k=min(k, len(gal_ids)))
        
        results = []
        for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), 1):
            gallery_id = gal_ids[idx.item()]
            result = {
                'rank': rank,
                'similarity_score': float(score),
                'folder_name': gallery_id,
                'chart_path': f"{gallery_id}/chart.png",
                'chart_type_parent': self.gallery_type_map.get(gallery_id, 'unknown'),
                'chart_type': self.gallery_raw_type_map.get(gallery_id, 'unknown')
            }
            results.append(result)
        
        return results
    
    def search(self, query: Dict, top_k: int = 10, **kwargs) -> Dict:
        """
        兼容 MultimodalRetriever.search() 的接口
        
        Args:
            query: 5-aspect dict 或其他查询
            top_k: 返回结果数量
            **kwargs: 兼容其他参数（如 chart_type_filter, layout_filter 等，当前忽略）
            
        Returns:
            Dict 格式与原来 MultimodalRetriever.search() 兼容
            {
                'query_info': {...},
                'results': [...],
                'search_stats': {...}
            }
        """
        import time
        start_time = time.time()
        
        # 执行检索
        results = self.retrieve(query, k=top_k)
        
        search_time = time.time() - start_time
        
        # 构造与原来兼容的返回格式
        response = {
            'query_info': {
                'type': '5_aspect_structured',
                'timestamp': start_time
            },
            'results': results,
            'search_stats': {
                'total_found': len(results),
                'search_time': round(search_time, 3),
                'filters_applied': {
                    'chart_type_filter': kwargs.get('chart_type_filter'),
                    'layout_filter': kwargs.get('layout_filter'),
                }
            }
        }
        
        return response


if __name__ == "__main__":
    retrieval = RetrievalV3()
    
    # 测试查询示例：完整 5-aspect 查询
    query = {
        "content": {"query": "sales data comparison across different quarters", "weight": 0.8},
        "style": {"query": "blue color scheme, professional look", "weight": 0.3},
        "layout": {"query": "vertical bars with clear labels", "weight": 0.2},
        "illustration": {"query": "", "weight": 0.0},
        "chart_type": {"query": "Bar Chart", "weight": 0.5}
    }
    
    print("\n" + "="*60)
    print("Test Query")
    print("="*60)
    print(json.dumps(query, indent=2, ensure_ascii=False))
    
    # 测试 retrieve 方法
    print("\n" + "="*60)
    print("Test retrieve() - Returns detailed results")
    print("="*60)
    results = retrieval.retrieve(query, k=10)
    print(f"\nTop-10 Retrieved Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['chart_path']} (type: {result['chart_type']}, score: {result['similarity_score']:.4f})")
    
    # 测试 search 方法（兼容旧接口）
    print("\n" + "="*60)
    print("Test search() - Compatible with MultimodalRetriever")
    print("="*60)
    search_result = retrieval.search(query, top_k=5)
    print(f"Total found: {search_result['search_stats']['total_found']}")
    print(f"Search time: {search_result['search_stats']['search_time']}s")
    for i, result in enumerate(search_result['results'], 1):
        print(f"  {i}. {result['chart_path']} (type: {result['chart_type']}, score: {result['similarity_score']:.4f})")
