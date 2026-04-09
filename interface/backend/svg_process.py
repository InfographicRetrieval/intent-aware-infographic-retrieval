import xml.etree.ElementTree as ET
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

class SVGStructureExtractor:
    def __init__(self, svg_path, image_counter_start=0, image_data_callback=None):
        """
        Args:
            svg_path: SVG 文件路径
            image_counter_start: 图片占位符起始编号（用于多SVG场景）
            image_data_callback: 回调函数，用于保存图片数据映射
                                 callback(placeholder, image_data, svg_path)
        """
        self.svg_path = svg_path
        self.tree = ET.parse(svg_path)
        self.root = self.tree.getroot()
        self.node_counter = 0
        self.node_mapping = {}
        
        # 图片处理相关
        self.image_counter = image_counter_start
        self.image_data_callback = image_data_callback
        self.image_replacements = {}  # 图片数据 -> 占位符的映射
        
        # 在初始化时预处理所有图片数据
        self._preprocess_images()
    
    def _preprocess_images(self):
        """在初始化时一次性提取所有图片数据并建立映射"""
        import re
        
        # 获取整个SVG的字符串表示
        svg_string = ET.tostring(self.root, encoding='unicode')
        
        # 查找所有超长的图片数据
        pattern = r'href="(data:image/[^"]{100,})"'
        matches = re.finditer(pattern, svg_string)
        
        # 为每个唯一的图片数据创建占位符
        seen_images = {}  # 图片数据 -> 占位符
        
        for match in matches:
            image_data = match.group(1)
            
            # 如果这个图片数据已经见过，复用占位符
            if image_data not in seen_images:
                self.image_counter += 1
                placeholder = f"[IMAGE_DATA_{self.image_counter}]"
                
                # 记录映射
                seen_images[image_data] = placeholder
                self.image_replacements[image_data] = placeholder
                
                # 回调通知外部
                if self.image_data_callback:
                    self.image_data_callback(placeholder, image_data, self.svg_path)
        
        # print(f"[DEBUG] 预处理完成，共发现 {len(self.image_replacements)} 个唯一图片")

    def get_global_defs(self):
        """提取SVG中的全局定义，包括 <style> 和 <defs> 中的内容"""
        global_defs = []
        
        # 查找所有 <style> 标签
        for style in self.root.findall('.//{http://www.w3.org/2000/svg}style'):
            style_code = ET.tostring(style, encoding='unicode')
            global_defs.append(style_code)
            
        # 查找 <defs> 标签并提取其子元素
        for defs in self.root.findall('.//{http://www.w3.org/2000/svg}defs'):
            for child in defs:
                child_code = ET.tostring(child, encoding='unicode')
                # 同样需要替换图片占位符
                for image_data, placeholder in self.image_replacements.items():
                    if image_data in child_code:
                        child_code = child_code.replace(f'href="{image_data}"', f'href="{placeholder}"')
                global_defs.append(child_code)
                
        return "\n".join(global_defs)
        
    def _get_element_svg_code(self, elem):
        """获取元素的SVG源码，使用预处理的图片映射进行替换"""
        svg_code = ET.tostring(elem, encoding='unicode')
        
        # 使用预处理好的映射进行替换
        for image_data, placeholder in self.image_replacements.items():
            if image_data in svg_code:
                svg_code = svg_code.replace(f'href="{image_data}"', f'href="{placeholder}"')
        
        return svg_code
    
    def get_final_image_counter(self):
        """返回处理完后的图片计数器值（用于下一个SVG）"""
        return self.image_counter
        
    def _clean_tag(self, tag):
        """移除命名空间前缀"""
        return tag.split('}')[-1] if '}' in tag else tag
    
    def _get_node_id(self):
        """生成唯一节点ID"""
        self.node_counter += 1
        return f"node_{self.node_counter}"
    
    def _extract_text_content(self, elem):
        """提取元素的文本内容"""
        texts = []
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
        for child in elem:
            if child.tail and child.tail.strip():
                texts.append(child.tail.strip())
        return ' '.join(texts) if texts else None
    
    def _get_element_signature(self, elem):
        """获取元素的结构签名（用于判断相似性）"""
        tag = self._clean_tag(elem.tag)
        children = list(elem)
        children_sig = ','.join([self._clean_tag(c.tag) for c in children])
        
        key_attrs = []
        for attr in ['class', 'id']:
            if elem.get(attr):
                key_attrs.append(f"{attr}={elem.get(attr)}")
        
        sig = f"{tag}[{children_sig}]"
        if key_attrs:
            sig += f"<{','.join(key_attrs)}>"
        
        return sig
    
    def _detect_repeating_pattern(self, children):
        """检测子元素中的重复模式"""
        if len(children) < 3:
            return None
        
        signatures = [self._get_element_signature(child) for child in children]
        sig_counts = Counter(signatures)
        
        for sig, count in sig_counts.items():
            if count >= 3:
                matching_elements = [
                    (i, child) for i, child in enumerate(children)
                    if self._get_element_signature(child) == sig
                ]
                return {
                    'signature': sig,
                    'count': count,
                    'elements': matching_elements,
                    'indices': [i for i, _ in matching_elements]
                }
        
        return None
    
    def _extract_important_values(self, elements, max_keep=5):
        """从重复元素中提取重要的值"""
        important = []
        
        # 保留首尾
        if len(elements) > 0:
            important.append(('first', 0, elements[0][1]))
        if len(elements) > 1:
            important.append(('last', len(elements)-1, elements[-1][1]))
        
        # 提取文本值找极值
        text_values = []
        for idx, (_, elem) in enumerate(elements):
            text = self._extract_text_content(elem)
            if text:
                try:
                    val = float(text.replace(',', ''))
                    text_values.append((idx, val, text, elem))
                except:
                    text_values.append((idx, None, text, elem))
        
        if text_values:
            numeric_values = [(i, v, t, e) for i, v, t, e in text_values if v is not None]
            if numeric_values:
                numeric_values.sort(key=lambda x: x[1])
                if numeric_values[0] not in [x[2] for x in important]:
                    important.append(('min_value', numeric_values[0][0], numeric_values[0][3]))
                if numeric_values[-1] not in [x[2] for x in important]:
                    important.append(('max_value', numeric_values[-1][0], numeric_values[-1][3]))
        
        # 均匀采样
        if len(elements) > 4:
            middle_indices = [len(elements) // 3, len(elements) * 2 // 3]
            for mid_idx in middle_indices:
                if len(important) < max_keep:
                    important.append(('sample', mid_idx, elements[mid_idx][1]))
        
        # 去重
        seen_indices = set()
        unique_important = []
        for reason, idx, elem in important:
            if idx not in seen_indices:
                seen_indices.add(idx)
                unique_important.append((reason, idx, elem))
                if len(unique_important) >= max_keep:
                    break
        
        return unique_important

    
    def _infer_semantic_role(self, elem):
        """推断元素的语义角色"""
        tag = self._clean_tag(elem.tag)
        elem_class = elem.get('class', '').lower()
        elem_id = elem.get('id', '').lower()
        combined = elem_class + ' ' + elem_id
        
        if 'axis' in combined or 'tick' in combined:
            return 'axis'
        elif 'legend' in combined:
            return 'legend'
        elif 'title' in combined or 'heading' in combined:
            return 'title'
        elif 'series' in combined or 'data' in combined:
            return 'data_series'
        elif 'grid' in combined:
            return 'grid'
        elif 'label' in combined:
            return 'label'
        elif 'bar' in combined:
            return 'bar_chart'
        elif tag == 'text':
            return 'text_element'
        elif tag in ['circle', 'rect', 'line', 'path']:
            return 'graphic_element'
        else:
            return 'container' if tag == 'g' else 'other'
    
    def _build_structure_tree(self, elem, depth=0, max_depth=10):
        """递归构建结构树"""
        if depth > max_depth:
            return {"type": "truncated", "reason": "max_depth_reached"}
        
        tag = self._clean_tag(elem.tag)
        node_id = self._get_node_id()
        children = list(elem)
        
        node = {
            'id': node_id,
            'tag': tag,
            'depth': depth
        }
        
        if elem.get('id'):
            node['svg_id'] = elem.get('id')
        if elem.get('class'):
            node['class'] = elem.get('class')
        
        # node['semantic_role'] = self._infer_semantic_role(elem)
        
        text = self._extract_text_content(elem)
        if text:
            node['text'] = text
        
        # 保存完整SVG代码
        svg_code = self._get_element_svg_code(elem)
        self.node_mapping[node_id] = {
            'tag': tag,
            'svg_id': elem.get('id'),
            'class': elem.get('class'),
            'attrib': dict(elem.attrib),
            'text': text,
            'svg_code': svg_code,
            'has_children': len(children) > 0
        }
        
        # 处理子元素
        if len(children) > 0:
            node['children_count'] = len(children)
            
            # 检测重复模式
            pattern = self._detect_repeating_pattern(children)
            
            if pattern and pattern['count'] >= 3:
                # 发现重复模式，进行压缩
                node['compressed'] = True
                node['pattern_info'] = {
                    'signature': pattern['signature'],
                    'total_count': pattern['count'],
                    'compressed_from': len(children)
                }
                
                # 只显示第一个完整元素
                first_element = pattern['elements'][0][1]
                node['first_element'] = self._build_structure_tree(first_element, depth + 1, max_depth)
                
                # 提取重要值（用于摘要显示）
                important = self._extract_important_values(pattern['elements'])
                node['important_examples'] = []
                for reason, idx, elem_child in important:
                    child_node = {
                        'index': idx,
                        'reason': reason,
                        'tag': self._clean_tag(elem_child.tag),
                    }
                    if elem_child.get('id'):
                        child_node['svg_id'] = elem_child.get('id')
                    if elem_child.get('class'):
                        child_node['class'] = elem_child.get('class')
                    
                    child_text = self._extract_text_content(elem_child)
                    if child_text:
                        child_node['text'] = child_text
                    
                    node['important_examples'].append(child_node)
                
                # 注意：不需要保存 all_children_code，因为每个子元素在递归时都会保存自己的 svg_code
                # 删除这行避免重复调用 _get_element_svg_code
                # self.node_mapping[node_id]['all_children_code'] = [
                #     self._get_element_svg_code(c) for c in children
                # ]
                
            else:
                # 没有重复模式，正常展开
                node['children'] = []
                for child in children:
                    child_node = self._build_structure_tree(child, depth + 1, max_depth)
                    node['children'].append(child_node)
        
        return node
    
    def generate_compressed_text(self, structure_data):
        """生成压缩后的文本表示"""
        
        def format_node(node, indent=0):
            """格式化单个节点"""
            prefix = "  " * indent
            lines = []
            
            tag = node['tag']
            
            desc_parts = [f"<{tag}>"]
            
            if node.get('svg_id'):
                desc_parts.append(f"id={node['svg_id']}")
            if node.get('class'):
                desc_parts.append(f"class={node['class']}")
            if node.get('semantic_role'):
                desc_parts.append(f"[{node['semantic_role']}]")
            
            desc = ' '.join(desc_parts)
            
            if node.get('text'):
                text = node['text'][:100] + '...' if len(node['text']) > 100 else node['text']
                desc += f' "{text}"'
            
            # 处理压缩的重复模式
            if node.get('compressed'):
                pattern = node['pattern_info']
                desc += f" [×{pattern['total_count']} similar] [SHOW:{node['id']}]"
                lines.append(f"{prefix}{desc}")
                
                # 显示第一个元素的完整内容
                lines.append(f"{prefix}  First element (pattern repeats {pattern['total_count']} times):")
                if node.get('first_element'):
                    first_lines = format_node(node['first_element'], indent + 2)
                    lines.extend(first_lines)
                
                # 显示重要值摘要
                if node.get('important_examples') and len(node['important_examples']) > 2:
                    lines.append(f"{prefix}  Value summary:")
                    for ex in node['important_examples']:
                        if ex.get('text'):
                            lines.append(f"{prefix}    [{ex['index']}] ({ex['reason']}): {ex['text']}")
                
            elif node.get('children'):
                # 有子节点但未压缩
                # desc += f" ({node['children_count']} children) [SHOW:{node['id']}]"
                desc += f"[SHOW:{node['id']}]"
                lines.append(f"{prefix}{desc}")
                
                # 递归显示子节点
                for child in node['children']:
                    lines.extend(format_node(child, indent + 1))
            
            else:
                # 叶节点
                desc += f" [SHOW:{node['id']}]"
                lines.append(f"{prefix}{desc}")
            
            return lines
        
        lines = [
            f"Root: {structure_data['root_attributes']}",
            "",
            "Compressed Tree:",
        ]
        
        lines.extend(format_node(structure_data['structure']))
        
        # lines.extend([
        #     "",
        #     "-" * 70,
        #     "Expansion Options:",
        #     "  [SHOW:node_id] - Show complete SVG source code for this node",
        #     "",
        #     "Note: Repeated patterns show the first element as template.",
        #     "      Use SHOW to see all instances."
        # ])
        
        return '\n'.join(lines)
    
    def extract(self):
        """提取SVG结构"""
        root_attrib = dict(self.root.attrib)
        structure_tree = self._build_structure_tree(self.root)
        
        return {
            'svg_path': str(self.svg_path),
            'svg_hash': hashlib.md5(ET.tostring(self.root)).hexdigest(),
            'root_attributes': root_attrib,
            'structure': structure_tree
        }
    
    def get_compressed_text(self):
        """获取压缩后的文本表示（不保存到文件）"""
        structure_data = self.extract()
        compressed_text = self.generate_compressed_text(structure_data)
        return compressed_text
    
    def get_node_mapping(self):
        """获取节点映射（用于展开）"""
        return self.node_mapping
    
    def save_outputs(self, output_dir):
        """保存所有输出"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        structure_data = self.extract()
        
        # 1. 保存完整数据
        full_data = {
            **structure_data,
            'node_mapping': self.node_mapping
        }
        
        with open(output_dir / 'structure_full.json', 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        # 2. 生成压缩文本
        compressed_text = self.generate_compressed_text(structure_data)
        
        with open(output_dir / 'compressed_structure.txt', 'w', encoding='utf-8') as f:
            f.write(compressed_text)
        
        # 3. 保存节点映射
        with open(output_dir / 'node_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.node_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved structure_full.json")
        print(f"✓ Saved compressed_structure.txt ({len(compressed_text)} chars)")
        print(f"✓ Saved node_mapping.json ({len(self.node_mapping)} nodes)")
        
        # 统计压缩效果
        original_size = len(ET.tostring(self.root, encoding='unicode'))
        compressed_size = len(compressed_text)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        print(f"\nCompression Stats:")
        print(f"  Original SVG: {original_size:,} chars")
        print(f"  Compressed: {compressed_size:,} chars")
        print(f"  Compression Ratio: {compression_ratio:.1f}x")
        
        return {
            'full_data': full_data,
            'compressed_text': compressed_text,
            'node_mapping': self.node_mapping
        }


# 使用示例
if __name__ == '__main__':
    svg_path = '/mnt/share/xujing/Chart_Retrieval/VizRetrieval/data/svg_converted/00000001/chart/chart.svg'
    output_dir = '/mnt/share/xujing/Chart_Retrieval/svg/output_v4'
    
    extractor = SVGStructureExtractor(svg_path)
    results = extractor.save_outputs(output_dir)
    
    print("\n" + "="*70)
    print("Compressed Text Preview:")
    print("="*70)
    print(results['compressed_text'])