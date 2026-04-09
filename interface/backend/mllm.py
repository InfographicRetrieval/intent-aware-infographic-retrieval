# mllm.py - Multimodal Large Language Model Handler
import re
import json
from PIL import Image
import io
import base64
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
from openai import OpenAI
import os
from svg_process import SVGStructureExtractor
from user_image_paths import resolve_user_image_path


class MLLM:
    """
    多模态大语言模型处理类
    支持文本和图像的多模态交互，包括SVG图表的压缩和展开
    """
    
    # 统一配置
    DEFAULT_MAX_TOKENS = 4096
    
    def __init__(self, llm_backend="Qwen/Qwen2.5-VL-72B-Instruct", max_tokens=None, external_placeholder_map: Optional[Dict[str, str]] = None):
        """
        初始化MLLM实例
        
        Args:
            llm_backend: 模型名称
            max_tokens: 最大生成token数，默认使用DEFAULT_MAX_TOKENS
        """
        # ===========================
        # 路径配置
        # ===========================
        self.base_dir = Path(__file__).parent.parent.parent / "interface_demo"
        
        # ===========================
        # API客户端配置
        # ===========================
        qwen_api_key = os.environ.get(
            "DASHSCOPE_API_KEY",
            os.environ.get("SILICONFLOW_API_KEY", ""),
        )
        qwen_base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            os.environ.get("SILICONFLOW_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        compat_api_key = os.environ.get(
            "OPENAI_API_KEY",
            os.environ.get(
                "CLOSEAI_API_KEY",
                os.environ.get("OPENAI_COMPAT_API_KEY", ""),
            ),
        )
        compat_base_url = os.environ.get(
            "OPENAI_BASE_URL",
            os.environ.get(
                "CLOSEAI_BASE_URL",
                os.environ.get("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"),
            ),
        )

        self.client_qwen = OpenAI(
            api_key=qwen_api_key or "DUMMY_KEY",
            base_url=qwen_base_url
        )

        self.client_others = OpenAI(
                api_key=compat_api_key or "DUMMY_KEY",
                base_url=compat_base_url
        )
        
        self.model_name = llm_backend
        self.max_tokens = max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
        
        # ===========================
        # SVG处理相关
        # ===========================
        self.svg_expanders = {}  # {svg_path: {'extractor': extractor, 'mapping': mapping}}
        self.current_svg_node_mapping = {}  # 当前对话中的SVG节点映射
        self.global_node_counter = 0
        
        # 图片占位符管理（统一占位符 Registry）
        # 重要：external_placeholder_map 若传入，必须是“会话级唯一真相源”，MLLM 将直接引用它（不复制）。
        self.image_placeholder_map = external_placeholder_map if external_placeholder_map is not None else {}  # {placeholder: 真实图片数据}
        self.svg_to_images = {}  # {svg_path: [placeholder_list]} 用于调试

        # 统一的全局图片计数器：从同一个 placeholder_map 推导，避免与 API 侧编号冲突
        self.global_image_counter = self._infer_next_image_counter(self.image_placeholder_map)
        
        # 根据模型名称选择对应的 client
        if "Qwen" in self.model_name or "qwen" in self.model_name:
            self.client = self.client_qwen
        else:
            self.client = self.client_others
        
        # ===========================
        # 加载System Prompts
        # ===========================
        self.system_prompt = []

        
        prompt_files = [
            "system_prompt_1.txt",
            "system_prompt_2.txt",
            "system_prompt_3.txt",
            "system_prompt_4.txt",
            "system_prompt_5.txt",
            "system_prompt_6.txt"
        ]
        
        for prompt_file in prompt_files:
            prompt_path = Path(__file__).parent / prompt_file
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt.append(f.read())

    @staticmethod
    def _infer_next_image_counter(placeholder_map: Dict[str, str]) -> int:
        """从现有的 [IMAGE_DATA_N] key 推导下一个可用编号（返回 next_id，起始为 0）。"""
        if not placeholder_map:
            return 0
        max_id = -1
        pattern = re.compile(r"\[IMAGE_DATA_(\d+)\]")
        for k in placeholder_map.keys():
            m = pattern.fullmatch(k.strip())
            if m:
                try:
                    max_id = max(max_id, int(m.group(1)))
                except ValueError:
                    pass
        return max_id + 1


    def clean_svg_code(self, svg_code: str, svg_path: str = None, use_compression: bool = True) -> str:
        """
        清理SVG代码，可以选择压缩或只替换图片数据
        
        Args:
            svg_code: SVG源代码
            svg_path: SVG文件路径
            use_compression: 是否使用压缩
            
        Returns:
            处理后的SVG代码或压缩文本
        """
        if use_compression and svg_path:
            import traceback
            print(f"[DEBUG] clean_svg_code 被调用: {svg_path}")
            # print("[DEBUG] 调用栈:")
            # for line in traceback.format_stack()[:-1]:
            #     print(line.strip())
            
            def save_image_data(placeholder, image_data, source_svg_path):
                """保存图片占位符映射"""
                self.image_placeholder_map[placeholder] = image_data
                
                if source_svg_path not in self.svg_to_images:
                    self.svg_to_images[source_svg_path] = []
                self.svg_to_images[source_svg_path].append(placeholder)
                
                print(f"[DEBUG] 保存图片映射: {placeholder} (来自 {source_svg_path})")

            extractor = SVGStructureExtractor(
                svg_path,
                image_counter_start=self.global_image_counter,
                image_data_callback=save_image_data
            )
            compressed_text = extractor.get_compressed_text()
            node_mapping = extractor.get_node_mapping()
            
            # 更新全局图片计数器
            self.global_image_counter = extractor.image_counter
            
            # 重新编号节点，确保全局唯一
            renumbered_mapping = {}
            node_id_map = {}
            
            for old_node_id in node_mapping.keys():
                self.global_node_counter += 1
                new_node_id = f"node_{self.global_node_counter}"
                node_id_map[old_node_id] = new_node_id
                renumbered_mapping[new_node_id] = node_mapping[old_node_id]
            
            # 替换压缩文本中的node ID
            renumbered_text = compressed_text
            for old_id, new_id in node_id_map.items():
                renumbered_text = renumbered_text.replace(f"[SHOW:{old_id}]", f"[SHOW:{new_id}]")
                renumbered_text = renumbered_text.replace(f"node_id: {old_id}", f"node_id: {new_id}")
            
            # 存储映射关系
            self.svg_expanders[svg_path] = {
                'extractor': extractor,
                'mapping': renumbered_mapping,
                'id_map': node_id_map
            }
            
            # 更新当前对话的节点映射
            self.current_svg_node_mapping.update(renumbered_mapping)
            
            return renumbered_text

        if not use_compression:
            # Legacy branch removed: keep SVG unchanged when compression is disabled
            return svg_code

    def convert_image_to_webp_base64(self, input_image_path: str) -> str:
        """
        将图片转换为webp格式的base64编码
        
        Args:
            input_image_path: 输入图片路径（可以是相对路径或绝对路径）
            
        Returns:
            base64编码的字符串，如果失败返回None
        """
        # 数据根目录（用于解析相对路径）
        DATA_ROOT = "/mnt/share/public/converted/converted"
        
        # 尝试直接使用路径
        full_path = input_image_path
        
        # 如果是相对路径，拼接数据根目录
        if not os.path.isabs(input_image_path) and not os.path.exists(input_image_path):
            full_path = os.path.join(DATA_ROOT, input_image_path)
        
        try:
            with Image.open(full_path) as img:
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='webp')
                byte_arr = byte_arr.getvalue()
                base64_str = base64.b64encode(byte_arr).decode('utf-8')
                return base64_str
        except IOError as e:
            print(f"Error: Unable to open or convert the image {input_image_path} (tried {full_path}): {e}")
            return None

    def create_multimodal_message(self, 
                                text_prompt: str, 
                                images: List[Union[str, Dict]] = None, 
                                reference_images: List[Union[str, Dict]] = None,
                                detail: str = "low",
                                stage: int = 0,
                                history: List[Dict] = None,
                                previous_query: str = None,
                                use_svg_compression: bool = True) -> Dict[str, Any]:
        """
        创建多模态消息格式
        
        Args:
            text_prompt: 文本提示
            images: 图片列表，可以是路径字符串、URL字符串或包含类型信息的字典
            reference_images: 参考图片列表
            detail: 图片详细程度，"low"或"high"
            stage: 阶段编号，用于选择system prompt
            history: 历史对话消息列表
            previous_query: 用于refinement的上一次检索查询
            use_svg_compression: 是否使用SVG压缩
            
        Returns:
            格式化的消息列表
        """
        content = []

        # Stage 5: 查询精炼的特殊处理
        if stage == 5:
            if not previous_query:
                raise ValueError("previous_query is required for stage 5 refinement.")

            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in (history or [])])
            
            user_prompt_text = (
                "Here is the context to refine the search query.\n\n"
                f"## Conversation History:\n---\n{formatted_history}\n---\n\n"
                f"## Previous Retrieval Query (that you generated):\n---\n{previous_query}\n---\n\n"
                f"## User's Refinement Input:\n---\n{text_prompt}\n---"
            )
            
            content.append({
                "type": "text",
                "text": user_prompt_text
            })
        
            if reference_images:
                for image in reference_images:
                    if isinstance(image, str):
                        svg_path = image.replace(".png", ".svg")
                        
                        print(f"检查SVG路径: {svg_path}")
                        if os.path.exists(svg_path):
                            try:
                                with open(svg_path, 'r', encoding='utf-8') as f:
                                    svg_code = f.read()
                                print(f"✓ SVG文件读取成功，长度: {len(svg_code)}")
                            except Exception as e:
                                print(f"✗ 无法读取SVG文件 {svg_path}: {e}")
                                raise ValueError(f"无法读取SVG文件 {svg_path}: {e}")
                        else:
                            print(f"⚠ SVG文件不存在: {svg_path}")
                            raise ValueError(f"SVG文件不存在: {svg_path}")
                        
                        # 添加图片
                        base64_image = self.convert_image_to_webp_base64(image)
                        if base64_image:
                            content.append({
                                "type": "text",
                                "text": f"READ-ONLY reference charts from retrieval (not the current editable SVG):"
                            })
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/webp;base64,{base64_image}",
                                    "detail": detail
                                }
                            })
                            
                            if svg_code:
                                try:
                                    if use_svg_compression:
                                        compressed_svg = self.clean_svg_code(
                                            svg_code,
                                            svg_path, 
                                            use_compression=True
                                        )
                                        content.append({
                                            "type": "text",
                                            "text": f"[Compressed SVG structure for the above chart:]\n{compressed_svg}"
                                        })
                                        print(f"✓ SVG压缩成功")
                                    else:
                                        raise ValueError(f"不压缩，只清理图片")
                                except Exception as e:
                                    print(f"✗ SVG处理失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    raise ValueError(f"SVG处理失败 {svg_path}: {e}")
                            else:
                                raise ValueError(f"svg code读取失败: {image}")
                                
                    elif isinstance(image, dict):
                        content.append(image)

        # 添加文本
        content.append({
            "type": "text",
            "text": text_prompt
        })

        # 处理用户上传的图片
        if images:
            for image in images:
                if isinstance(image, str):
                    if image.startswith(('http://', 'https://')):
                        # URL形式
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": detail
                            }
                        })
                    else:
                        # 本地路径，转换为base64
                        base64_image = self.convert_image_to_webp_base64(image)
                        if base64_image:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/webp;base64,{base64_image}",
                                    "detail": detail
                                }
                            })
                elif isinstance(image, dict):
                    content.append(image)
        
        # 初始化消息列表和最新retrieval图片列表
        messages = []
        latest_retrieval_images = []
        
        # 添加历史对话
        if history:
            print("length of history:", len(history))
            recent_history = history
            
            # 找到最新一次retrieval的图片
            for msg in reversed(recent_history):
                if msg["role"] == "assistant" and isinstance(msg["content"], list):
                    assistant_images = []
                    for item in msg["content"]:
                        if item["type"] == "image_reference":
                            assistant_images.append(item["image_path"])
                    
                    if assistant_images:
                        latest_retrieval_images = assistant_images
                        break
            
            # 处理历史消息
            for msg in recent_history:
                if msg["role"] == "user":
                    user_content = []
                    
                    user_content.append({
                        "type": "text",
                        "text": msg["content"]
                    })
                    
                    # 如果有用户上传的图片，添加到消息中
                    if "user_image_path" in msg and msg["user_image_path"]:
                        full_user_image_path = resolve_user_image_path(msg["user_image_path"])
                        base64_image = self.convert_image_to_webp_base64(full_user_image_path) if full_user_image_path else None
                        if base64_image:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/webp;base64,{base64_image}",
                                    "detail": detail
                                }
                            })
                    
                    messages.append({
                        "role": msg["role"],
                        "content": user_content if len(user_content) > 1 else msg["content"]
                    })
                elif msg["role"] == "assistant":
                    if isinstance(msg["content"], list):
                        # 新格式：提取文本部分
                        assistant_text = ""
                        for item in msg["content"]:
                            if item["type"] == "text":
                                assistant_text = item["text"]
                                break
                        
                        messages.append({
                            "role": msg["role"],
                            "content": assistant_text
                        })
                    else:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
        # 构建当前用户消息，可能包含参考图片
        current_user_content = []
        
        # 优先使用传入的 reference_images 参数（用于 Stage 1/2/3）
        # 如果没有传入，则从历史中提取（用于多轮对话）
        images_to_use = reference_images if reference_images else latest_retrieval_images
        
        if images_to_use:
            current_user_content.append({
                "type": "text",
                "text": f"[Reference charts from retrieval system ({len(images_to_use)} images), which are READ-ONLY reference:]"
            })
            
            for idx, image in enumerate(images_to_use):
                if isinstance(image, str):
                    # 处理图片路径
                    base64_image = self.convert_image_to_webp_base64(image)
                    if base64_image:
                        current_user_content.append({
                            "type": "text",
                            "text": f"[Reference Image {idx + 1}:]"
                        })
                        current_user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{base64_image}",
                                "detail": detail
                            }
                        })
                        
                        # 添加 SVG 代码
                        svg_path = image.replace(".png", ".svg")
                        
                        # 处理相对路径：与 convert_image_to_webp_base64 使用相同的逻辑
                        DATA_ROOT = "/mnt/share/public/converted/converted"
                        if not os.path.isabs(svg_path) and not os.path.exists(svg_path):
                            full_svg_path = os.path.join(DATA_ROOT, svg_path)
                        else:
                            full_svg_path = svg_path
                        
                        if os.path.exists(full_svg_path):
                            try:
                                with open(full_svg_path, 'r', encoding='utf-8') as f:
                                    svg_code = f.read()
                                
                                if use_svg_compression:
                                    processed_svg = self.clean_svg_code(
                                        svg_code, 
                                        full_svg_path, 
                                        use_compression=True
                                    )
                                    current_user_content.append({
                                        "type": "text",
                                        "text": f"[Compressed SVG structure for Reference Image {idx + 1}:]\n{processed_svg}"
                                    })
                            except Exception as e:
                                print(f"[DEBUG] ✗ SVG处理失败 {full_svg_path}: {e}")
                                import traceback
                                traceback.print_exc()
                elif isinstance(image, dict):
                    current_user_content.append(image)
            
            current_user_content.append({
                "type": "text",
                "text": "[Current user question and user uploaded charts, which should not be regarded as reference images:]"
            })
        
        # 添加用户的原始内容
        if stage != 5:
            current_user_content.extend(content)
        elif images:
            current_user_content.extend(content[1:])

        # 添加完整的当前用户消息
        messages.append({"role": "user", "content": current_user_content})
        
        # 在消息列表开头插入 system prompt
        messages.insert(0, {"role": "system", "content": self.system_prompt[stage]})
        
        print("selected_stage:", stage)
        
        # 调试输出
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                print(item["text"])
                            elif item.get("type") == "image_url":
                                print("A IMAGE\n")
                else:
                    print(msg["content"])
        
        return messages

    def create_base64_message(self, text_prompt: str, image_path: str, detail: str = "low") -> Dict[str, Any]:
        """
        创建base64形式的消息格式
        
        Args:
            text_prompt: 文本提示
            image_path: 图片路径
            detail: 图片详细程度
            
        Returns:
            格式化的消息字典
        """
        base64_image = self.convert_image_to_webp_base64(image_path)
        if not base64_image:
            raise ValueError(f"无法转换图片: {image_path}")
            
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{base64_image}",
                        "detail": detail
                    }
                },
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
        }

    def show_full_svg(self, node_id: str) -> str:
        """
        展开指定节点的完整SVG代码（图片数据已被占位符替换）
        
        Args:
            node_id: 节点ID
            
        Returns:
            完整的SVG代码
        """
        if node_id not in self.current_svg_node_mapping:
            return f"Error: Node {node_id} not found in current SVG mapping"
        
        mapping = self.current_svg_node_mapping[node_id]
        svg_code = mapping['svg_code']
        
        # 统计占位符数量
        import re
        placeholders = re.findall(r'\[IMAGE_DATA_\d+\]', svg_code)
        
        result = f"Full SVG code for {node_id}:\n{svg_code}\n"
        
        if placeholders:
            result += f"\nNote: This code contains {len(placeholders)} image placeholder(s): {', '.join(set(placeholders))}"
            result += "\n      These placeholders will be automatically filled with actual image data during rendering."
            result += "\n      Keep them as-is in your output - DO NOT modify or remove them."
        
        return result

    def restore_image_placeholders(self, svg_code: str) -> str:
        """
        将所有 [IMAGE_DATA_N] 占位符替换回真实的图片数据
        
        Args:
            svg_code: 包含占位符的SVG代码
            
        Returns:
            恢复了图片数据的SVG代码
        """
        if '[IMAGE_DATA_' not in svg_code:
            return svg_code
        
        import re
        placeholders = re.findall(r'\[IMAGE_DATA_(\d+)\]', svg_code)
        unique_placeholders = sorted(set(placeholders), key=lambda x: int(x), reverse=True)
        
        for num in unique_placeholders:
            placeholder = f"[IMAGE_DATA_{num}]"
            if placeholder in self.image_placeholder_map:
                real_data = self.image_placeholder_map[placeholder]
                svg_code = svg_code.replace(placeholder, real_data)
                print(f"[DEBUG] 恢复图片数据: {placeholder} ({len(real_data)} chars)")
            else:
                print(f"[WARNING] 未找到占位符映射: {placeholder}")
        
        return svg_code

    def get_svg_tools_definition(self) -> list:
        """
        获取SVG工具的定义（用于function calling）
        
        Returns:
            工具定义列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "show_full_svg",
                    "description": "Show the complete SVG source code for a specific node. Use this when you need to see the detailed SVG code of a compressed node.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "The node ID to expand (e.g., 'node_1', 'node_9'). You can find available node IDs marked as [SHOW:node_id] in the compressed structure."
                            }
                        },
                        "required": ["node_id"]
                    }
                }
            }
        ]

    def send_message(self, message: Dict[str, Any], use_tools: bool = False) -> Union[str, Dict]:
        """
        发送消息到API
        
        Args:
            message: 格式化的message
            use_tools: 是否使用工具（function calling）
            
        Returns:
            API响应或包含工具调用信息的字典
        """
        if not self.client:
            return "模拟响应: 消息已发送，但未配置真实API客户端"
        
        print("using model:", self.model_name)
        
        try:
            api_params = {
                "model": self.model_name,
                "messages": message,
            }
            
            if use_tools:
                api_params["tools"] = self.get_svg_tools_definition()
                api_params["tool_choice"] = "auto"
            
            if "gpt" in self.model_name:
                api_params["reasoning_effort"] = "low"
            
            response = self.client.chat.completions.create(**api_params)
            
            # 获取 token 使用量
            usage = getattr(response, 'usage', None)
            token_usage = {
                'prompt_tokens': usage.prompt_tokens if usage else 0,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': usage.total_tokens if usage else 0
            }
            print(f"[Token Usage] input: {token_usage['prompt_tokens']}, output: {token_usage['completion_tokens']}, total: {token_usage['total_tokens']}")
            
            message_response = response.choices[0].message
            
            if use_tools and message_response.tool_calls:
                return {
                    "type": "tool_call",
                    "message": message_response,
                    "tool_calls": message_response.tool_calls,
                    "usage": token_usage
                }
            else:
                return {
                    "content": message_response.content,
                    "usage": token_usage
                }
                
        except Exception as e:
            return f"API调用错误: {str(e)}"

    def handle_tool_calls(self, messages: list, tool_calls: list, previous_usage: Dict = None) -> Dict:
        """
        处理工具调用并获取最终响应
        支持多轮 tool call，直到模型不再调用工具为止
        
        Args:
            messages: 当前的消息列表
            tool_calls: 工具调用列表
            previous_usage: 之前调用的 token 使用量
            
        Returns:
            包含最终文本响应和累计 token 使用量的字典
        """
        total_usage = previous_usage or {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        current_tool_calls = tool_calls
        round_num = 0
        
        while current_tool_calls:
            round_num += 1
            print(f"[Tool Call Round {round_num}] Processing {len(current_tool_calls)} tool call(s)")
            
            # 执行当前轮次的工具调用
            for tool_call in current_tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"[Tool Call] {function_name}({arguments})")
                
                if function_name == "show_full_svg":
                    result = self.show_full_svg(arguments.get('node_id'))
                else:
                    result = f"Unknown function: {function_name}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # 调用 API 获取下一轮响应
            if "gpt" in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.get_svg_tools_definition(),
                    tool_choice="auto",
                    reasoning_effort="low"
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.get_svg_tools_definition(),
                    tool_choice="auto"
                )
            
            # 累计 token 使用量
            usage = getattr(response, 'usage', None)
            if usage:
                total_usage['prompt_tokens'] += usage.prompt_tokens
                total_usage['completion_tokens'] += usage.completion_tokens
                total_usage['total_tokens'] += usage.total_tokens
            
            print(f"[Token Usage - Round {round_num}] input: {usage.prompt_tokens if usage else 0}, output: {usage.completion_tokens if usage else 0}, total: {usage.total_tokens if usage else 0}")
            
            message_response = response.choices[0].message
            
            # 检查是否还有新的 tool_calls
            if message_response.tool_calls:
                # 添加 assistant 的 tool_call 消息到历史
                messages.append(message_response)
                current_tool_calls = message_response.tool_calls
                print(f"[Tool Call Round {round_num}] Model requested another round with {len(current_tool_calls)} tool call(s)")
            else:
                # 没有更多 tool calls，返回最终内容
                print(f"[Tool Call Round {round_num}] No more tool calls, returning final response")
                print(f"[Token Usage - Cumulative] input: {total_usage['prompt_tokens']}, output: {total_usage['completion_tokens']}, total: {total_usage['total_tokens']}")
                
                return {
                    "content": message_response.content,
                    "usage": total_usage
                }
        
        # 理论上不会到达这里，但为了安全返回空结果
        return {
            "content": "No tool calls processed",
            "usage": total_usage
        }

    def parse_retrieval_query(self, response: str) -> Optional[Dict]:
        """
        从模型输出中提取 <retrieval>{JSON}</retrieval> 并解析为 5-aspect 结构
        
        Args:
            response: 模型响应文本
            
        Returns:
            包含 5 个 aspect 的字典，解析失败返回 None
            格式: {
                "content": {"query": str, "weight": float},
                "style": {"query": str, "weight": float},
                "layout": {"query": str, "weight": float},
                "illustration": {"query": str, "weight": float},
                "chart_type": {"query": str, "weight": float}
            }
        """
        match = re.search(r"<retrieval>(.*?)</retrieval>", response, re.DOTALL)
        if not match:
            return None
        
        try:
            data = json.loads(match.group(1).strip())
            
            # 验证必需字段
            required_aspects = ['content', 'style', 'layout', 'illustration', 'chart_type']
            for aspect in required_aspects:
                if aspect not in data:
                    return None
                # 确保每个 aspect 有 query 和 weight
                if not isinstance(data[aspect], dict):
                    return None
                data[aspect]['query'] = data[aspect].get('query', '')
                data[aspect]['weight'] = float(data[aspect].get('weight', 0.0))
            
            return data
            
        except (json.JSONDecodeError, ValueError):
            return None
    
    def parse_chart_types(self, response: str) -> List[str]:
        """
        从模型输出中提取 <chart_types>xxx</chart_types> 中间的内容
        
        Args:
            response: 模型响应文本
            
        Returns:
            提取的chart type列表，如果没有找到则返回空列表
        """
        match = re.search(r"<chart_types>(.*?)</chart_types>", response, re.DOTALL)
        if not match:
            return []
        
        content = match.group(1).strip()
        # 按逗号分割并清理空白
        types = [t.strip() for t in content.split(',') if t.strip()]
        return types

    def parse_select(self, response: str) -> list:
        """
        从模型输出中提取 <select>xxx</select> 中间的内容
        
        Args:
            response: 模型响应文本
            
        Returns:
            选择的索引列表
        """
        match = re.search(r"<select>(.*?)</select>", response, re.DOTALL).group(1).strip()
        try:
            match_list = json.loads(match)
        except:
            match_list = None
        print(match_list)
        return match_list

    def parse_output(self, response: str, restore_placeholders: bool = False) -> str:
        """
        从模型输出中提取 <output>xxx</output> 中间的内容。

        注意：
        - 默认不展开 [IMAGE_DATA_N]，避免把 base64 写进对话历史。
        - 如果你确实要返回给前端一个可直接渲染的 SVG（包含 base64），才传 restore_placeholders=True。
        """
        output_content = ""
        try:
            match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
            if match:
                output_content = match.group(1).strip()
            else:
                output_content = response
        except Exception:
            output_content = response

        if restore_placeholders:
            return self.restore_image_placeholders(output_content)
        return output_content


if __name__ == "__main__":
    # 测试代码
    mllm = MLLM()
    
    message = mllm.create_base64_message(
        "请描述这张图片的内容", 
        "/mnt/share/xujing/Chart_Retrieval/VizRetrieval/data/svg_converted/00000000/chart/convert_chart.png"
    )
    print(message)
    response = mllm.send_message(message)
    print(response)
