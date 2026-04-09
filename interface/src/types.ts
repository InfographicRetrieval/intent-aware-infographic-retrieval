export interface ImageCandidate {
  chart_path: string;
  chart_type: string;
  chart_type_parent?: string | null;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export type SelectionMode = 'auto' | 'manual' | 'direct';
export type RetrievalAspectKey = 'chart_type' | 'content' | 'layout' | 'style' | 'illustration';

export interface RetrievalAspectValue {
  query: string;
  weight: number;
}

export type RetrievalQuerySpec = Record<RetrievalAspectKey, RetrievalAspectValue>;

export interface HistoryTextContentPart {
  type: 'text';
  text: string;
}

export interface HistoryImageReferencePart {
  type: 'image_reference';
  image_path: string;
}

export type HistoryContentPart =
  | HistoryTextContentPart
  | HistoryImageReferencePart
  | {
      type: string;
      [key: string]: unknown;
    };

export type HistoryMessageContent = string | HistoryContentPart[];

export interface ChatResponse {
  retrieval_query?: string;
  output?: string;
  image_gallery: ImageCandidate[];
  used_history: boolean;
  stage: string;
  needs_user_selection?: boolean;
  chart_type_filter?: string[];
  candidate_count?: number;
  suggested_chart_types?: string[];
  token_usage?: TokenUsage;
  svg_placeholder_map?: Record<string, string>;
}

export interface FinalResponse {
  output: string;
  selected_gallery: string[];
  selection_mode: SelectionMode;
  used_history: boolean;
  stage: string;
  chart_type_filter?: string[];
  candidate_count?: number;
  retrieval_query?: string;
  image_gallery?: string[];
  token_usage?: TokenUsage;
  svg_placeholder_map?: Record<string, string>;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'selection';
  content: string;
  image?: File | null;
  userImagePath?: string;
  retrieval_query?: string;
  image_gallery?: ImageCandidate[] | string[];  // 支持新旧格式
  selected_images?: string[];  // 存储选中的图片路径
  used_history?: boolean;
  timestamp: Date;
  isLoading?: boolean;
  showRetrievalQuery?: boolean;
  showImageGallery?: boolean;
  showOutput?: boolean;
  needsSelection?: boolean;
  originalQuery?: string;
  chart_type_filter?: string[];
  candidate_count?: number;
  token_usage?: TokenUsage;
  accumulated_token_usage?: TokenUsage;  // 用于累计不显示阶段的 token 使用量
  selected_gallery?: string[];
  selection_mode?: SelectionMode;
}

export interface ChatSession {
  id: string;
  name: string;
  createdAt: Date;
  lastMessageAt: Date;
  messageCount: number;
}

export interface ChartTypeNode {
  name: string;
  path: string;
  level: number;
  chart_types: string[];
  count: number;
  percentage: number;
  children: ChartTypeNode[];
}

export interface ChartTypeHierarchy {
  hierarchy: ChartTypeNode[];
  flat_mapping: { [key: string]: { path: string[]; ancestors: string[] } };
  statistics: {
    total_roots: number;
    total_types_mapped: number;
    total_types_original: number;
    missing_types: string[];
  };
  default_selection: string[];
}
