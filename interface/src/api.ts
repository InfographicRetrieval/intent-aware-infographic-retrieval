import {
  ChatResponse,
  FinalResponse,
  ChatSession,
  ChartTypeHierarchy,
  HistoryMessageContent,
  ImageCandidate,
  SelectionMode,
} from './types';

// API返回的 session 对象，日期是字符串格式
export interface ApiChatSession extends Omit<ChatSession, 'createdAt' | 'lastMessageAt'> {
  createdAt: string;
  lastMessageAt: string;
}

// API返回的历史消息对象
export interface ApiHistoryMessage {
  role: 'user' | 'assistant';
  content: HistoryMessageContent;
  timestamp: string;
  selected_gallery?: string[];
  image_gallery?: ImageCandidate[] | string[];
  retrieval_query?: string;
  user_image_path?: string;
  chart_type_filter?: string[];
  candidate_count?: number;
  selection_mode?: SelectionMode;
}

export interface SessionHistoryResponse {
  session_id: string;
  history: ApiHistoryMessage[];
  svg_placeholder_map: Record<string, string>;
}

// 当前登录Username
let currentUsername: string = localStorage.getItem('chart_retrieval_username') || '';

export const setUsername = (username: string) => {
  currentUsername = username;
  localStorage.setItem('chart_retrieval_username', username);
};

export const getUsername = (): string => {
  return currentUsername;
};

export const clearUsername = () => {
  currentUsername = '';
  localStorage.removeItem('chart_retrieval_username');
};

// 统一的响应处理函数
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const errorText = await response.text();
    console.error("API Error Response:", errorText);
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

export const getSessions = async (): Promise<{ sessions: ApiChatSession[] }> => {
  const response = await fetch(`/api/sessions?username=${encodeURIComponent(currentUsername)}`);
  return handleResponse(response);
};

export const getSessionHistory = async (sessionId: string): Promise<SessionHistoryResponse> => {
  const response = await fetch(`/api/chat/history/${sessionId}?username=${encodeURIComponent(currentUsername)}`);
  return handleResponse(response);
};

export const getChartTypes = async (): Promise<ChartTypeHierarchy> => {
  const response = await fetch('/api/chart-types');
  return handleResponse(response);
};

export const deleteSessionById = async (sessionId: string): Promise<Response> => {
  const response = await fetch(`/api/chat/history/${sessionId}?username=${encodeURIComponent(currentUsername)}`, { 
    method: 'DELETE' 
  });
  if (!response.ok) {
     const errorText = await response.text();
     console.error("API Error Response:", errorText);
     throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  return response;
};

export const copySessionById = async (sessionId: string): Promise<{ session_id: string }> => {
  const response = await fetch(`/api/chat/history/${sessionId}/copy?username=${encodeURIComponent(currentUsername)}`, {
    method: 'POST',
  });
  return handleResponse(response);
};

export const deleteLastTurnBySessionId = async (sessionId: string): Promise<{ session_id: string; remaining_messages: number }> => {
  const response = await fetch(`/api/chat/history/${sessionId}/last-turn?username=${encodeURIComponent(currentUsername)}`, {
    method: 'DELETE',
  });
  return handleResponse(response);
};

export const postChatMessage = async (formData: FormData): Promise<ChatResponse> => {
  // 添加 username 到 formData
  formData.append('username', currentUsername);
  const response = await fetch('/api/chat', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
};

export const finalizeChatSelection = async (formData: FormData): Promise<FinalResponse> => {
  // 添加 username 到 formData
  formData.append('username', currentUsername);
  const response = await fetch('/api/chat/finalize', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
};

export const refineChatQuery = async (formData: FormData): Promise<ChatResponse> => {
  // 添加 username 到 formData
  formData.append('username', currentUsername);
  const response = await fetch('/api/chat/refine', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
};

export const reretrieveChatQuery = async (formData: FormData): Promise<ChatResponse> => {
  formData.append('username', currentUsername);
  const response = await fetch('/api/chat/reretrieve', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
};

// Baseline 检索（简易版）
export interface BaselineSearchResult {
  rank: number;
  similarity_score: number;
  folder_name: string;
  chart_path: string;
  chart_type: string;
}

export const baselineSearch = async (query: string, top_k: number = 10): Promise<{ query: string; results: BaselineSearchResult[]; total: number }> => {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('top_k', String(top_k));
  
  const response = await fetch('/api/baseline/search', {
    method: 'POST',
    body: formData,
  });
  return handleResponse(response);
};
