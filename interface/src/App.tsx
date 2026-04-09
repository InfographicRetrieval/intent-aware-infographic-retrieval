import React, { useState, useRef, useEffect } from 'react';
import * as api from './api';
import type { ApiHistoryMessage } from './api';

import {
  ChatMessage,
  ChatSession,
  ImageCandidate,
  ChartTypeNode,
  ChartTypeHierarchy,
  HistoryContentPart,
  RetrievalQuerySpec,
} from './types';

import { LoginScreen, ChartTypeTreeNode, RetrievalQueryTable, MarkdownRenderer } from './chatComponents';
import ReferencePanel from './components/ReferencePanel';
import {
  extractAllSvgCodes,
  findChartTypeNodeByName,
  getAllChildNodeNames,
} from './utils/appHelpers';
import {
  parseRetrievalSpec,
  retrievalSpecsEqual,
  serializeRetrievalSpec,
} from './utils/retrievalQuery';

const UserImageAttachment: React.FC<{ file?: File | null; storedPath?: string }> = ({ file, storedPath }) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(storedPath ? `/api/image/${storedPath}` : null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file, storedPath]);

  if (!previewUrl) return null;

  return (
    <img
      src={previewUrl}
      alt={file ? 'Uploaded file preview' : 'Uploaded file'}
      className="max-w-xs rounded-md"
    />
  );
};

const extractHistoryMessageContent = (msg: ApiHistoryMessage): {
  content: string;
  imageGallery?: ImageCandidate[] | string[];
} => {
  const isTextPart = (item: HistoryContentPart): item is HistoryContentPart & { type: 'text'; text: string } =>
    item.type === 'text' && typeof item.text === 'string';
  const isImageReferencePart = (
    item: HistoryContentPart
  ): item is HistoryContentPart & { type: 'image_reference'; image_path: string } =>
    item.type === 'image_reference' && typeof item.image_path === 'string';

  let content = msg.content;
  let imageGallery = msg.selected_gallery || msg.image_gallery;

  if (msg.role === 'assistant' && Array.isArray(msg.content)) {
    const textContent = msg.content.find(isTextPart);
    const imageReferences = msg.content.filter(isImageReferencePart);

    content = textContent?.text || '';
    if (imageReferences.length > 0) {
      imageGallery = imageReferences.map((item) => item.image_path);
    }
  }

  return {
    content: typeof content === 'string' ? content : '',
    imageGallery,
  };
};

const App: React.FC = () => {
  const [userInput, setUserInput] = useState<string>('');
  const [userImage, setUserImage] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  
  // 用户登录状态
  const [username, setUsername] = useState<string>(() => api.getUsername());
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(() => !!api.getUsername());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => {
    const newSessionId = Date.now().toString();
    return newSessionId;
  });
  const [enlargedImage, setEnlargedImage] = useState<string | null>(null);
  const [chartTypeHierarchy, setChartTypeHierarchy] = useState<ChartTypeNode[]>([]);
  const [pendingSelection, setPendingSelection] = useState<{
    messageId: string;
    userText: string;
    userImage: File | null;
    imageGallery: ImageCandidate[];
    retrievalQuery?: string;
    candidateCount?: number;
    accumulatedTokenUsage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  } | null>(null);
  const [editedRetrievalSpec, setEditedRetrievalSpec] = useState<RetrievalQuerySpec | null>(null);
  const [reretrieveLoading, setReretrieveLoading] = useState<boolean>(false);
  const [selectedImages, setSelectedImages] = useState<string[]>([]);
  // 用于Search结果图片选择界面的chart type过滤（存储节点名称）
  const [selectionChartTypeFilter, setSelectionChartTypeFilter] = useState<Set<string>>(new Set());
  const [selectionExpandedNodes, setSelectionExpandedNodes] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState<number>(0);
  const [selectedModel, setSelectedModel] = useState<string>('gpt-5.4');
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [referencePanelOpen, setReferencePanelOpen] = useState<boolean>(true);
  const [referencePanelWidth, setReferencePanelWidth] = useState<number>(320); // 默认320px
  const [isResizing, setIsResizing] = useState<boolean>(false);
  const [currentReferenceImages, setCurrentReferenceImages] = useState<string[]>([]);
  const [draftUserImageUrl, setDraftUserImageUrl] = useState<string | null>(null);
  // 存储所有生成过的 SVG 版本（从 assistant 消息中的 ```svg``` 代码块提取）
  const [generatedSvgs, setGeneratedSvgs] = useState<Array<{ key: string; messageId: string; timestamp: Date; svgCode: string; pngUrl?: string }>>([]);
  const [svgPlaceholderMap, setSvgPlaceholderMap] = useState<Record<string, string>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sessionLoadSeqRef = useRef(0);

  // 可用的模型选项
  const modelOptions = [
    { value: 'gpt-5.4', label: 'GPT-5.4' },
    { value: 'gpt-5.1', label: 'GPT-5.1'},
    { value: 'gpt-5-mini-2025-08-07', label: 'GPT-5 Mini' },
    { value: 'Qwen/Qwen2.5-VL-72B-Instruct', label: 'Qwen2.5-VL-72B' },
    { value: 'Pro/Qwen/Qwen2.5-VL-7B-Instruct', label: 'Qwen2.5-VL-7B' }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 监听 messages 变化：提取所有 SVG 版本
  useEffect(() => {
    setGeneratedSvgs(extractAllSvgCodes(messages));
  }, [messages]);

  useEffect(() => {
    if (!userImage) {
      setDraftUserImageUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(userImage);
    setDraftUserImageUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [userImage]);

  // 监听 messages 变化，自动更新 reference images
  useEffect(() => {
    // 从最新的消息开始往前查找，找到最近一次有 image_gallery 的 assistant 消息
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.type === 'assistant' && msg.image_gallery && msg.image_gallery.length > 0) {
        // 处理新旧格式兼容
        const imagePaths = typeof msg.image_gallery[0] === 'string' 
          ? msg.image_gallery as string[]
          : (msg.image_gallery as ImageCandidate[]).map(img => img.chart_path);
        setCurrentReferenceImages(imagePaths);
        return; // 找到后立即返回
      }
    }
    // 如果没有找到任何 reference images，Clear状态
    setCurrentReferenceImages([]);
  }, [messages]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      
      const newWidth = window.innerWidth - e.clientX;
      // 限制宽度在 250px 到 600px 之间
      if (newWidth >= 250 && newWidth <= 600) {
        setReferencePanelWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // 登录后加载 sessions/chart types（避免首次未登录时加载空Username数据）
  useEffect(() => {
    if (!isLoggedIn || !username) return;
    loadSessions();
    loadChartTypes();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, username]);

  const getCurrentAppliedRetrievalSpec = (): RetrievalQuerySpec | null =>
    parseRetrievalSpec(pendingSelection?.retrievalQuery);

  const loadSessionPlaceholderMap = async (sessionId: string): Promise<Record<string, string>> => {
    const data = await api.getSessionHistory(sessionId);
    return data.svg_placeholder_map || {};
  };

  // 从后端加载sessions
  const loadSessions = async () => {
    try {
      const data = await api.getSessions(); // 使用新函数
      if (data.sessions && data.sessions.length > 0) {
        const convertedSessions = data.sessions.map((session: any) => ({
          ...session,
          createdAt: new Date(session.createdAt),
          lastMessageAt: new Date(session.lastMessageAt)
        }));
        setSessions(convertedSessions);
        // 如果当前session不在列表中，切换到第一个session
        const currentSessionExists = convertedSessions.find((s: ChatSession) => s.id === currentSessionId);
        if (!currentSessionExists) {
          const firstSession = convertedSessions[0];
          setCurrentSessionId(firstSession.id);
          // 异步加载第一个session的消息
          setTimeout(() => switchSession(firstSession.id), 0);
        } else {
          // 如果当前session存在，加载其消息
          setTimeout(() => switchSession(currentSessionId), 0);
        }
      } else {
        // 如果没有sessions，创建默认session
        const defaultSession: ChatSession = {
          id: currentSessionId,
          name: 'New Chat',
          createdAt: new Date(),
          lastMessageAt: new Date(),
          messageCount: 0
        };
        setSessions([defaultSession]);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
      // 创建默认session
      const defaultSession: ChatSession = {
        id: currentSessionId,
        name: 'New Chat',
        createdAt: new Date(),
        lastMessageAt: new Date(),
        messageCount: 0
      };
      setSessions([defaultSession]);
    }
  };

  const loadChartTypes = async () => {
    try {
      const data: ChartTypeHierarchy = await api.getChartTypes();
      setChartTypeHierarchy(data.hierarchy || []);
    } catch (error) {
      console.error('Error loading chart types:', error);
      setChartTypeHierarchy([]);
    }
  };



  // 创建新session
  const createNewSession = () => {
    const newSessionId = Date.now().toString();
    const newSession: ChatSession = {
      id: newSessionId,
      name: 'New Chat',
      createdAt: new Date(),
      lastMessageAt: new Date(),
      messageCount: 0
    };
    setGeneratedSvgs([]);
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSessionId);
    setMessages([]);
    setGeneratedSvgs([]);
    setSvgPlaceholderMap({});
    setPendingSelection(null);
    setEditedRetrievalSpec(null);
    setSelectedImages([]);
    setSelectionChartTypeFilter(new Set());
    setSelectionExpandedNodes(new Set());
    setCurrentPage(0);
  };

  const switchSession = async (sessionId: string, force: boolean = false) => {
    if (!force && sessionId === currentSessionId) return;

    const seq = ++sessionLoadSeqRef.current;

    setCurrentSessionId(sessionId);
    setPendingSelection(null);
    setEditedRetrievalSpec(null);
    setSelectedImages([]);

    // ✅ 切换时先清理旧 session 的渲染缓存，避免串数据
    setGeneratedSvgs([]);

    try {
      const data = await api.getSessionHistory(sessionId);

      // ✅ 防止请求乱序：只接受最后一次切换的结果
      if (seq !== sessionLoadSeqRef.current) return;

      setSvgPlaceholderMap(data.svg_placeholder_map || {});

      if (data.history) {
        const convertedMessages: ChatMessage[] = data.history.map((msg: ApiHistoryMessage, index: number) => {
          const { content, imageGallery } = extractHistoryMessageContent(msg);

          return {
            id: `${sessionId}-${index}`,
            type: msg.role === 'user' ? 'user' : 'assistant',
            content: content,
            timestamp: new Date(msg.timestamp),
            retrieval_query: msg.retrieval_query,
            image_gallery: imageGallery,
            used_history: msg.role === 'assistant' ? true : undefined,
            userImagePath: msg.user_image_path,
            chart_type_filter: msg.chart_type_filter,
            candidate_count: msg.candidate_count,
            selection_mode: msg.selection_mode
          };
        });

        setMessages(convertedMessages);
      } else {
        setMessages([]);
      }
    } catch (error) {
      console.error('Error loading session history:', error);
      if (seq !== sessionLoadSeqRef.current) return;
      setMessages([]);
      setSvgPlaceholderMap({});
    }
  };


  // 删除session
  const deleteSession = async (sessionId: string) => {
    if (sessions.length <= 1) return; // 至少保留一个session
    
    try {
      await api.deleteSessionById(sessionId); // 使用新函数
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      
      if (sessionId === currentSessionId) {
        // 如果删除的是当前session，切换到第一个可用session
        const remainingSessions = sessions.filter(s => s.id !== sessionId);
        if (remainingSessions.length > 0) {
          switchSession(remainingSessions[0].id);
        }
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  const copySession = async (sessionId: string) => {
    try {
      const data = await api.copySessionById(sessionId);
      await loadSessions();
      await switchSession(data.session_id);
    } catch (error) {
      console.error('Error copying session:', error);
    }
  };

  const deleteLastTurn = async () => {
    if (!currentSessionId || messages.length === 0) return;

    try {
      await api.deleteLastTurnBySessionId(currentSessionId);
      setPendingSelection(null);
      setEditedRetrievalSpec(null);
      setSelectedImages([]);
      setSelectionChartTypeFilter(new Set());
      setSelectionExpandedNodes(new Set());
      setCurrentPage(0);
      await loadSessions();
      await switchSession(currentSessionId, true);
    } catch (error) {
      console.error('Error deleting last turn:', error);
      alert('Failed to delete the last turn. Please try again.');
    }
  };

  // 更新session信息
  const updateSessionInfo = (sessionId: string, messageCount: number) => {
    setSessions(prev => prev.map(session => 
      session.id === sessionId 
        ? { 
            ...session, 
            lastMessageAt: new Date(), 
            messageCount,
            name: messageCount === 1 ? messages[0]?.content.slice(0, 20) + '...' || 'New Chat' : session.name
          }
        : session
    ));
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUserImage(file);
    }
  };

const handleSubmit = async () => {
  if (!userInput.trim()) {
    alert('Please enter text content');
    return;
  }
      // 检查是否为优化Search的步骤
      if (pendingSelection) {
        const refinementQuery = (pendingSelection.retrievalQuery || '').trim();

        // 兜底：如果没有可用的 previous_query，退出 refine 状态并按新Search处理
        if (!refinementQuery) {
          setPendingSelection(null);
          setEditedRetrievalSpec(null);
          setSelectedImages([]);
          setSelectionChartTypeFilter(new Set());
          setSelectionExpandedNodes(new Set());
          setCurrentPage(0);
          setMessages(prev =>
            prev.map(msg =>
              msg.id === pendingSelection.messageId
                ? {
                    ...msg,
                    needsSelection: false,
                    content: 'The previous retrieval selection was not completed. That selection flow has been closed.'
                  }
                : msg
            )
          );
        } else {
          // 步骤 1: 立即将用户的输入消息添加到UI，以实现即时反馈
          const userMessage: ChatMessage = {
            id: Date.now().toString(),
            type: 'user',
            content: userInput,
            image: userImage,
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, userMessage]);
          setLoading(true);
      
          // 步骤 2: 保存当前输入用于API调用，然后Clear输入框
          const currentInput = userInput;
          const currentImage = userImage;
          setUserInput('');
          setUserImage(null);
      
          try {
            const formData = new FormData();
            formData.append('session_id', currentSessionId);
            formData.append('refinement_text', currentInput);
            formData.append('previous_query', refinementQuery);
            formData.append('model_name', selectedModel);
            // 不再传递chart_types，后端返回所有候选图片
            if (currentImage) {
              formData.append('user_image', currentImage);
            }
    
          // 步骤 3: 调用新的refine API
          const result = await api.refineChatQuery(formData);
    
        const updatedCandidateCount = result.candidate_count ?? result.image_gallery.length;

          // 步骤 4: API返回后，只更新旧的选择消息中的图片画廊，不再重复添加用户消息
          setMessages(prev =>
            prev.map(msg => {
              if (msg.id === pendingSelection.messageId) {
                return {
                  ...msg,
                  retrieval_query: result.retrieval_query,
                  image_gallery: result.image_gallery,
                  candidate_count: updatedCandidateCount,
                  content: `Found ${updatedCandidateCount} relevant images. Please select the ones you need:`
                };
              }
              return msg;
            })
          );

        setSelectedImages([]);
          
          // 步骤 5: 更新待选择项的状态
          setPendingSelection(prev =>
            prev
              ? {
                  ...prev,
                  userText: currentInput,
                  userImage: currentImage,
                  imageGallery: result.image_gallery,
                  retrievalQuery: result.retrieval_query,
                  candidateCount: updatedCandidateCount,
                }
              : null
          );
          setEditedRetrievalSpec(parseRetrievalSpec(result.retrieval_query));
          
          // 初始化选择界面：不再受 suggested_chart_types 影响，仅根据 image_gallery 构建筛选树
          setSelectionChartTypeFilter(new Set());
          setSelectionExpandedNodes(new Set());
          setCurrentPage(0);
    
        } catch (error) {
          console.error('Error refining query:', error);
          const errorMessage: ChatMessage = {
            id: (Date.now() + 1).toString(),
            type: 'assistant',
            content: 'An error occurred while refining retrieval. Please try again.',
            timestamp: new Date(),
          };
          setMessages(prev => [...prev, errorMessage]);
          } finally {
            setLoading(false);
          }
      
          return; // 结束优化流程
        }
      }
  

    // 添加用户消息
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: userInput,
      image: userImage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    // Clear输入
    const currentInput = userInput;
    const currentImage = userImage;
    setUserInput('');
    setUserImage(null);

    try {
      const formData = new FormData();
      formData.append('user_text', currentInput);
      formData.append('session_id', currentSessionId);
      formData.append('model_name', selectedModel);
      // 不再在Search时传递chart_types，后端返回所有候选图片
      if (currentImage) {
        formData.append('user_image', currentImage);
      }

      const result = await api.postChatMessage(formData); // 使用新函数

      if (result.stage === 'image_selection' && result.needs_user_selection) {
        // 第一阶段：显示Search结果，等待用户选择
        const candidateCount = result.candidate_count ?? result.image_gallery.length;
        const selectionMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'selection',
          content: `Found ${candidateCount} relevant images. Please select the ones you need:`,
          retrieval_query: result.retrieval_query,
          image_gallery: result.image_gallery,
          used_history: result.used_history,
          timestamp: new Date(),
          needsSelection: true,
          originalQuery: currentInput,
          candidate_count: candidateCount,
          accumulated_token_usage: result.token_usage  // 保存累计 token 使用量（从这个阶段开始）
        };
        
        setMessages(prev => [...prev, selectionMessage]);
        
        setSelectedImages([]);
        // 保存待处理的选择信息
        setPendingSelection({
          messageId: selectionMessage.id,
          userText: currentInput,
          userImage: currentImage,
          imageGallery: result.image_gallery,
          retrievalQuery: result.retrieval_query,
          candidateCount,
          accumulatedTokenUsage: result.token_usage  // 保存累计 token
        });
        setEditedRetrievalSpec(parseRetrievalSpec(result.retrieval_query));
        
        // 初始化选择界面：不再受 suggested_chart_types 影响，仅根据 image_gallery 构建筛选树
        setSelectionChartTypeFilter(new Set());
        setSelectionExpandedNodes(new Set());
        setCurrentPage(0);
        
      } else if (result.stage === 'completed') {
        try {
          const latestPlaceholderMap = await loadSessionPlaceholderMap(currentSessionId);
          setSvgPlaceholderMap(latestPlaceholderMap);
        } catch (placeholderError) {
          console.error('Error loading placeholder map for completed response:', placeholderError);
        }
        // 直接完成（无需Search）
        const assistantMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: result.output || 'Sorry, no response was generated',
          retrieval_query: result.retrieval_query,
          image_gallery: result.image_gallery,
          used_history: result.used_history,
          timestamp: new Date(),
          chart_type_filter: result.chart_type_filter,
          candidate_count: result.candidate_count,
          token_usage: result.token_usage
        };
        
        setMessages(prev => {
          const newMessages = [...prev, assistantMessage];
          updateSessionInfo(currentSessionId, newMessages.length);
          return newMessages;
        });
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'An error occurred. Please try again',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSvgPlaceholderMap({});
    setPendingSelection(null);
    setEditedRetrievalSpec(null);
    setSelectedImages([]);
    setSelectionChartTypeFilter(new Set());
    setSelectionExpandedNodes(new Set());
    setCurrentPage(0);
  };

  const handleImageSelection = (imagePath: string) => {
    setSelectedImages(prev => {
      if (prev.includes(imagePath)) {
        return prev.filter(p => p !== imagePath);
      } else {
        return [...prev, imagePath];
      }
    });
  };

  const applyAdjustedRetrievalSpec = async () => {
    if (!pendingSelection || !editedRetrievalSpec) return;

    setReretrieveLoading(true);
    try {
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      formData.append('retrieval_query', serializeRetrievalSpec(editedRetrievalSpec));

      const result = await api.reretrieveChatQuery(formData);
      const updatedCandidateCount = result.candidate_count ?? result.image_gallery.length;

      setMessages(prev =>
        prev.map(msg => {
          if (msg.id !== pendingSelection.messageId) return msg;
          return {
            ...msg,
            retrieval_query: result.retrieval_query,
            image_gallery: result.image_gallery,
            candidate_count: updatedCandidateCount,
            content: `Found ${updatedCandidateCount} relevant images. Please select the ones you need:`
          };
        })
      );

      setPendingSelection(prev =>
        prev
          ? {
              ...prev,
              imageGallery: result.image_gallery,
              retrievalQuery: result.retrieval_query,
              candidateCount: updatedCandidateCount,
            }
          : null
      );
      setEditedRetrievalSpec(parseRetrievalSpec(result.retrieval_query) || editedRetrievalSpec);
      setSelectedImages([]);
      setSelectionChartTypeFilter(new Set());
      setSelectionExpandedNodes(new Set());
      setCurrentPage(0);
    } catch (error) {
      console.error('Error re-retrieving with adjusted spec:', error);
      alert('Failed to re-retrieve with the adjusted spec. Please try again.');
    } finally {
      setReretrieveLoading(false);
    }
  };


  const submitSelection = async (mode: 'auto' | 'manual' | 'direct') => {
    if (!pendingSelection) return;
    
    setLoading(true);
    
    try {
      const formData = new FormData();
      formData.append('user_text', pendingSelection.userText);
      formData.append('session_id', currentSessionId);
      formData.append('selection_mode', mode);
      formData.append('model_name', selectedModel);
      
      if (pendingSelection.userImage) {
        formData.append('user_image', pendingSelection.userImage);
      }
      
      if (pendingSelection.retrievalQuery) {
        formData.append('retrieval_query', pendingSelection.retrievalQuery);
      }

      // 不再传递chart_types到后端
      formData.append(
        'candidate_count',
        String(pendingSelection.candidateCount ?? pendingSelection.imageGallery.length)
      );
      
      // 根据选择模式决定传递的图片
      if (mode === 'direct') {
        // 直接回答模式，不需要图片
        formData.append('selected_images', JSON.stringify([]));
      } else if (mode === 'auto') {
        // Auto模式：直接选前10个图片，不再按chart type过滤
        const filteredGallery = pendingSelection.imageGallery.slice(0, 10);  // 只取前10个
        const imagePaths = filteredGallery.map((img: ImageCandidate) => img.chart_path);
        formData.append('selected_images', JSON.stringify(imagePaths));
      } else {
        // Manual模式：使用用户选中的图片，不进行AI精排
        if (selectedImages.length === 0) {
          alert('Please select at least one image');
          setLoading(false);
          return;
        }
        // selectedImages 现在直接存储图片路径
        formData.append('selected_images', JSON.stringify(selectedImages));
      }

      
      const result = await api.finalizeChatSelection(formData); // 使用新函数
      try {
        const latestPlaceholderMap = await loadSessionPlaceholderMap(currentSessionId);
        setSvgPlaceholderMap(latestPlaceholderMap);
      } catch (placeholderError) {
        console.error('Error loading placeholder map after finalize:', placeholderError);
      }
      
      
      // 累计 token 使用量（阶段 1 + 阶段 2）
      const previousUsage = pendingSelection.accumulatedTokenUsage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
      const currentUsage = result.token_usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
      const accumulatedUsage = {
        prompt_tokens: previousUsage.prompt_tokens + currentUsage.prompt_tokens,
        completion_tokens: previousUsage.completion_tokens + currentUsage.completion_tokens,
        total_tokens: previousUsage.total_tokens + currentUsage.total_tokens
      };
      
      // 更新选择消息，显示用户的选择
      setMessages(prev => prev.map(msg => {
        if (msg.id === pendingSelection.messageId) {
          let selectionInfo: string[] | undefined;
          if (mode === 'manual') {
            // Manual模式：使用用户手动选择的图片
            selectionInfo = selectedImages;
          } else if (mode === 'auto') {
            // Auto模式：使用AI精排后选择的图片
            selectionInfo = result.selected_gallery;
          } else if (mode === 'direct') {
            selectionInfo = undefined;
          } else {
            selectionInfo = [];
          }
          
          return {
            ...msg,
            selected_images: selectionInfo,
            needsSelection: false,
            candidate_count: pendingSelection.candidateCount ?? pendingSelection.imageGallery.length,
            selection_mode: mode
          };
        }
        return msg;
      }));
      
      // 添加最终回答
      const finalMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: result.output,
        image_gallery: result.selected_gallery,
        used_history: result.used_history,
        timestamp: new Date(),
        chart_type_filter: result.chart_type_filter,
        candidate_count: result.candidate_count,
        retrieval_query: result.retrieval_query,
        token_usage: accumulatedUsage  // 显示累计的 token 使用量
      };
      
      setMessages(prev => {
        const newMessages = [...prev, finalMessage];
        updateSessionInfo(currentSessionId, newMessages.length);
        return newMessages;
      });
      
      // 清理状态
      setPendingSelection(null);
      setEditedRetrievalSpec(null);
      setSelectedImages([]);
      
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: 'An error occurred while generating the final response. Please try again',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  // 登录处理
  const handleLogin = (user: string) => {
    api.setUsername(user);
    setUsername(user);
    setIsLoggedIn(true);
  };
  
  // 登出处理
  const handleLogout = () => {
    api.clearUsername();
    setGeneratedSvgs([]);
    setSvgPlaceholderMap({});
    setUsername('');
    setIsLoggedIn(false);
    setSessions([]);
    setMessages([]);
    setCurrentSessionId(Date.now().toString());
    setEditedRetrievalSpec(null);
  };
  
  // 如果未登录，显示登录界面
  if (!isLoggedIn) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* 侧边栏 */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden bg-gray-900 text-white flex flex-col`}>
        {/* 侧边栏头部 */}
        <div className="p-4 border-b border-gray-700">
          {/* 用户信息 */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-bold">
                {username.charAt(0).toUpperCase()}
              </div>
              <span className="text-sm font-medium truncate max-w-[100px]">{username}</span>
            </div>
            <button
              onClick={handleLogout}
              className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
              title="Log out"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Chat History</h2>
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-1 hover:bg-gray-700 rounded"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <button
            onClick={createNewSession}
            className="w-full mt-3 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-md text-sm font-medium transition-colors flex items-center justify-center space-x-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>

        {/* 会话列表 */}
        <div className="flex-1 overflow-y-auto p-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group relative p-3 mb-2 rounded-lg cursor-pointer transition-colors ${
                session.id === currentSessionId
                  ? 'bg-blue-600 text-white'
                  : 'hover:bg-gray-700 text-gray-300'
              }`}
              onClick={() => switchSession(session.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-medium truncate">{session.name}</h3>
                  <p className="text-xs opacity-75 mt-1">
                    {session.messageCount} messages · {session.lastMessageAt.toLocaleDateString()}
                  </p>
                </div>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      copySession(session.id);
                    }}
                    className="p-1 hover:bg-gray-600 rounded"
                    title="Copy session"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h6a2 2 0 002-2v-8a2 2 0 00-2-2h-6a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </button>
                  {sessions.length > 1 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSession(session.id);
                      }}
                      className="p-1 hover:bg-red-600 rounded"
                      title="Delete session"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 flex flex-row">
        {/* 左侧主要内容区域 */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* 标题栏 */}
          <div className="bg-white border-b border-gray-200 px-6 py-4">
            <div className="flex justify-between items-center">
              <div className="flex items-center space-x-4">
                {!sidebarOpen && (
                  <button
                    onClick={() => setSidebarOpen(true)}
                    className="p-2 hover:bg-gray-100 rounded-md"
                  >
                    <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                  </button>
                )}
                <div>
                  <h1 className="text-xl font-bold text-gray-900">🧠 ChartRetriever</h1>
                  <p className="text-sm text-gray-600">chart retrieval system enhanced MLLMs</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={deleteLastTurn}
                  disabled={messages.length === 0 || loading}
                  className="px-4 py-2 text-sm bg-amber-50 text-amber-700 rounded-md hover:bg-amber-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Delete last turn
                </button>
                <button
                  onClick={clearChat}
                  className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
                >
                  Clear current conversation
                </button>
              </div>
            </div>
          </div>

      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-20">
              <div className="text-6xl mb-4">💬</div>
              <p className="text-lg">Start your conversation!</p>
              <p className="text-sm mt-2">You can send text messages or upload images for multimodal interaction</p>
            </div>
          ) : (
            messages.map((message) => (
            <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl ${message.type === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200'} rounded-lg p-4 shadow-sm`}>
                {/* 消息头部 */}
                <div className="flex items-center mb-2">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${message.type === 'user' ? 'bg-blue-500' : message.type === 'selection' ? 'bg-orange-200 text-orange-600' : 'bg-gray-200 text-gray-600'}`}>
                    {message.type === 'user' ? '👤' : message.type === 'selection' ? '🎯' : '🤖'}
                  </div>
                  <span className={`ml-2 text-xs ${message.type === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                  {message.type === 'assistant' && message.used_history
                  //  && (
                    // <span className="ml-2 px-2 py-1 text-xs bg-green-100 text-green-700 rounded-full">
                    //   📚 使用历史上下文
                    // </span>
                  // )
                  }
                </div>

                {/* 用户图片 */}
                {(message.image || message.userImagePath) && (
                  <div className="mb-3">
                    <UserImageAttachment file={message.image} storedPath={message.userImagePath} />
                  </div>
                )}

                {/* 消息内容 */}
                <div className={`${message.type === 'user' ? 'text-white' : 'text-gray-800'}`}>
                  {message.type === 'user' ? (
                    <pre className="whitespace-pre-wrap font-sans">{message.content}</pre>
                  ) : (
                    <div className="prose prose-sm max-w-none">
                      <MarkdownRenderer content={message.content} svgPlaceholderMap={svgPlaceholderMap} />
                    </div>
                  )}
                </div>

                {/* Search查询（仅助手消息） */}
                {message.type === 'assistant' && message.retrieval_query && (
                  <div className="mt-3 p-3 bg-gray-50 rounded-md border-l-4 border-blue-400">
                    <RetrievalQueryTable query={message.retrieval_query} />
                  </div>
                )}

                {/* Search查询（选择阶段） */}
                {message.type === 'selection' && message.retrieval_query && (
                  <div className="mt-3 p-3 bg-blue-50 rounded-md border-l-4 border-blue-400">
                    <RetrievalQueryTable
                      query={message.retrieval_query}
                      isSelection
                      editable={pendingSelection?.messageId === message.id}
                      editedSpec={pendingSelection?.messageId === message.id ? editedRetrievalSpec : null}
                      onEditedSpecChange={pendingSelection?.messageId === message.id ? setEditedRetrievalSpec : undefined}
                      onApplyChanges={pendingSelection?.messageId === message.id ? applyAdjustedRetrievalSpec : undefined}
                      applyDisabled={
                        pendingSelection?.messageId !== message.id ||
                        reretrieveLoading ||
                        !editedRetrievalSpec ||
                        retrievalSpecsEqual(editedRetrievalSpec, getCurrentAppliedRetrievalSpec())
                      }
                      isApplying={pendingSelection?.messageId === message.id && reretrieveLoading}
                    />
                  </div>
                )}

                {/* 图片选择界面 */}
                {message.type === 'selection' && message.needsSelection && (() => {
                  const gallery = (message.image_gallery || []) as ImageCandidate[];

                  const findFallbackBasicChild = (root: ChartTypeNode): string | undefined =>
                    root.children.find((child) => child.name.toLowerCase().startsWith('basic '))?.name;

                  const resolveSelectionChartType = (img: ImageCandidate): { parentName?: string; childName?: string } => {
                    const rootFromParent = img.chart_type_parent
                      ? chartTypeHierarchy.find((root) => root.name === img.chart_type_parent)
                      : undefined;

                    if (rootFromParent) {
                      if (rootFromParent.children.some((child) => child.name === img.chart_type)) {
                        return { parentName: rootFromParent.name, childName: img.chart_type };
                      }
                      if (img.chart_type === rootFromParent.name) {
                        return {
                          parentName: rootFromParent.name,
                          childName: findFallbackBasicChild(rootFromParent),
                        };
                      }
                      if (rootFromParent.chart_types.includes(img.chart_type)) {
                        return { parentName: rootFromParent.name };
                      }
                    }

                    for (const root of chartTypeHierarchy) {
                      if (root.name === img.chart_type) {
                        return {
                          parentName: root.name,
                          childName: findFallbackBasicChild(root),
                        };
                      }
                      if (root.children.some((child) => child.name === img.chart_type)) {
                        return { parentName: root.name, childName: img.chart_type };
                      }
                      if (root.chart_types.includes(img.chart_type)) {
                        return { parentName: root.name };
                      }
                    }

                    return {
                      parentName: img.chart_type_parent || undefined,
                      childName: img.chart_type || undefined,
                    };
                  };

                  const normalizedGallery = gallery.map((img) => ({
                    image: img,
                    ...resolveSelectionChartType(img),
                  }));
                  
                  // 两层结构匹配：chart_type_parent 是外层，chart_type 是内层
                  const parentCounts: {[key: string]: number} = {};
                  const childCountsByParent: {[parent: string]: {[child: string]: number}} = {};

                  normalizedGallery.forEach(({ parentName, childName }) => {
                    if (!parentName) return;

                    parentCounts[parentName] = (parentCounts[parentName] || 0) + 1;
                    if (!childName) return;
                    if (!childCountsByParent[parentName]) childCountsByParent[parentName] = {};
                    childCountsByParent[parentName][childName] = (childCountsByParent[parentName][childName] || 0) + 1;
                  });

                  // 仅按两层结构构建过滤树：父类 -> 子类
                  const buildFilteredHierarchy = (): ChartTypeNode[] => {
                    const filteredRoots: ChartTypeNode[] = [];

                    chartTypeHierarchy.forEach((root) => {
                      const rootCount = parentCounts[root.name] || 0;
                      if (rootCount === 0) return;

                      const childrenCountMap = childCountsByParent[root.name] || {};
                      const filteredChildren = root.children
                        .filter((child) => (childrenCountMap[child.name] || 0) > 0)
                        .map((child) => ({
                          ...child,
                          children: [],
                          count: childrenCountMap[child.name] || 0
                        }));

                      filteredRoots.push({
                        ...root,
                        children: filteredChildren,
                        count: rootCount
                      });
                    });

                    return filteredRoots;
                  };

                  const filteredHierarchy = buildFilteredHierarchy();

                  // 自动展开所有根节点（仅当展开状态为空时）
                  if (selectionExpandedNodes.size === 0 && filteredHierarchy.length > 0) {
                    const rootNames = filteredHierarchy.map(root => root.name);
                    setSelectionExpandedNodes(new Set(rootNames));
                  }

                  // 仅按两层规则过滤：选父类命中 chart_type_parent，选子类命中 chart_type
                  const selectedParentNames = new Set<string>();
                  const selectedChildNames = new Set<string>();
                  filteredHierarchy.forEach((root) => {
                    if (selectionChartTypeFilter.has(root.name)) {
                      selectedParentNames.add(root.name);
                    }
                    root.children.forEach((child) => {
                      if (selectionChartTypeFilter.has(child.name)) {
                        selectedChildNames.add(child.name);
                      }
                    });
                  });

                  const filteredGallery = normalizedGallery
                    .filter(({ parentName, childName }) => {
                    if (selectionChartTypeFilter.size === 0) return true;
                    const matchesParent = !!parentName && selectedParentNames.has(parentName);
                    const matchesChild = !!childName && selectedChildNames.has(childName);
                    return matchesParent || matchesChild;
                    })
                    .map(({ image }) => image);

                  // 分页
                  const imagesPerPage = 6;
                  const totalPages = Math.ceil(filteredGallery.length / imagesPerPage);
                  const startIndex = currentPage * imagesPerPage;
                  const endIndex = startIndex + imagesPerPage;
                  const currentImages = filteredGallery.slice(startIndex, endIndex);
                  
                  // 切换节点选择（在选择界面）
                  const toggleSelectionNode = (nodeName: string) => {
                    setSelectionChartTypeFilter(prev => {
                      const newSelected = new Set(prev);
                      const isCurrentlySelected = newSelected.has(nodeName);
                      
                      const node = findChartTypeNodeByName(chartTypeHierarchy, nodeName);
                      if (!node) return prev;
                      
                      const allRelatedNodes = getAllChildNodeNames(node);
                      
                      if (isCurrentlySelected) {
                        allRelatedNodes.forEach(name => newSelected.delete(name));
                      } else {
                        allRelatedNodes.forEach(name => newSelected.add(name));
                      }
                      
                      return newSelected;
                    });
                    setSelectedImages([]);
                    setCurrentPage(0);
                  };
                  
                  // 切换展开状态
                  const toggleSelectionExpansion = (nodeName: string) => {
                    setSelectionExpandedNodes(prev => {
                      const newExpanded = new Set(prev);
                      if (newExpanded.has(nodeName)) {
                        newExpanded.delete(nodeName);
                      } else {
                        newExpanded.add(nodeName);
                      }
                      return newExpanded;
                    });
                  };
                  
                  return (
                  <div className="mt-4">
                    {/* Chart Type 树形过滤器 */}
                    <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-sm font-medium text-gray-800">Filter chart types:</h4>
                        <div className="text-xs text-gray-600">
                          Showing {filteredGallery.length} / {gallery.length}
                        </div>
                      </div>
                      <div className="bg-white rounded border border-gray-200 p-2">
                        {filteredHierarchy.map((rootNode) => (
                          <ChartTypeTreeNode
                            key={rootNode.name}
                            node={rootNode}
                            selectedNodes={selectionChartTypeFilter}
                            expandedNodes={selectionExpandedNodes}
                            onToggleSelect={toggleSelectionNode}
                            onToggleExpand={toggleSelectionExpansion}
                            level={0}
                          />
                        ))}
                      </div>
                      <div className="mt-2 flex justify-between items-center">
                        <div className="text-xs text-gray-600">
                          {selectionChartTypeFilter.size} categories selected
                        </div>
                        <div className="flex space-x-2">
                          <button
                            onClick={() => {
                              const allNodes = new Set<string>();
                              const collect = (nodes: ChartTypeNode[]) => {
                                nodes.forEach(n => {
                                  allNodes.add(n.name);
                                  collect(n.children);
                                });
                              };
                              collect(filteredHierarchy);
                              setSelectionChartTypeFilter(allNodes);
                              setSelectedImages([]);
                              setCurrentPage(0);
                            }}
                            className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                          >
                            Select All
                          </button>
                          <button
                            onClick={() => {
                              setSelectionChartTypeFilter(new Set());
                              setSelectedImages([]);
                              setCurrentPage(0);
                            }}
                            className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                          >
                            Clear
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* 显示过滤后的图片 */}
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-800 mb-3">
                        Please select the images you need (multi-select supported, click to enlarge):
                      </h4>
                      {filteredGallery.length === 0 ? (
                        <div className="text-center text-gray-500 py-8">
                          No images match the current filter
                        </div>
                      ) : (
                        <>
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {currentImages.map((imageCandidate: ImageCandidate, displayIndex: number) => {
                              return (
                              <div key={imageCandidate.chart_path} className="relative">
                                <div 
                                  className={`border-2 rounded-lg overflow-hidden transition-all ${
                                    selectedImages.includes(imageCandidate.chart_path) 
                                      ? 'border-blue-500 bg-blue-50 shadow-md' 
                                      : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                                  }`}
                                >
                                  <img
                                    src={`/api/image/${encodeURIComponent(imageCandidate.chart_path)}`}
                                    alt="Reference"
                                    className="w-full h-48 object-contain bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
                                    onClick={() => setEnlargedImage(`/api/image/${encodeURIComponent(imageCandidate.chart_path)}`)}
                                    onError={(e) => {
                                      const target = e.target as HTMLImageElement;
                                      target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvueJh+WKoOi9veWksei0pTwvdGV4dD48L3N2Zz4=';
                                    }}
                                  />
                                  {/* 选择按钮 */}
                                  <div className="absolute top-2 right-2">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleImageSelection(imageCandidate.chart_path);
                                      }}
                                      className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                                        selectedImages.includes(imageCandidate.chart_path)
                                          ? 'bg-blue-500 text-white shadow-lg'
                                          : 'bg-white text-gray-600 border border-gray-300 hover:bg-gray-50'
                                      }`}
                                    >
                                      {selectedImages.includes(imageCandidate.chart_path) ? '✓' : '+'}
                                    </button>
                                  </div>
                                  {/* Chart Type 标签 */}
                                  <div className="absolute top-2 left-2">
                                    <span className="px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded-full font-medium shadow-sm">
                                      {imageCandidate.chart_type}
                                    </span>
                                  </div>
                                </div>
                                <div className="p-2 bg-white">
                                  <p className="text-xs text-gray-500 truncate" title={imageCandidate.chart_path}>
                                    #{startIndex + displayIndex + 1}
                                  </p>
                                  <p className="text-xs text-blue-600 mt-1">Click the image to enlarge</p>
                                </div>
                              </div>
                            )})}
                          </div>
                          
                          {/* 分页控件 */}
                          {totalPages > 1 && (
                            <div className="flex items-center justify-center mt-4 space-x-2">
                              <button
                                onClick={() => setCurrentPage(prev => Math.max(0, prev - 1))}
                                disabled={currentPage === 0}
                                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Previous
                              </button>
                              <span className="text-sm text-gray-600">
                                Page {currentPage + 1} / {totalPages}
                              </span>
                              <button
                                onClick={() => setCurrentPage(prev => Math.min(totalPages - 1, prev + 1))}
                                disabled={currentPage === totalPages - 1}
                                className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                Next
                              </button>
                            </div>
                          )}
                        </>
                      )}
                      {selectedImages.length > 0 && (
                        <p className="text-sm text-blue-600 mt-3 font-medium">
                          ✅ {selectedImages.length} images selected
                        </p>
                      )}
                    </div>
                    
                    {/* 操作按钮 - 只在需要选择时显示 */}
                    <div className="pt-4 border-t border-gray-200">
                      <div className="text-xs text-gray-500 text-center mb-2">
                        You have selected {selectedImages.length} images. You can also continue without selecting any.
                      </div>
                      <div className="text-xs text-gray-500 text-center mb-4">
                        If you're not satisfied with current results, add more requirements in the input below and we'll continue refining based on this retrieval round.
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <button
                          onClick={() => submitSelection('auto')}
                          disabled={loading}
                          title="AI Auto-Rerank: the system selects images first, then answers (recommended for beginners)"
                          aria-describedby="action-auto-desc"
                          className="relative w-full text-left p-4 rounded-lg border border-blue-300 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors shadow-sm"
                        >
                          <span className="absolute top-2 right-2 px-2 py-0.5 text-[10px] font-semibold bg-white text-blue-700 rounded-full">
                            Recommended
                          </span>
                          <div className="text-sm font-semibold text-white mb-1">🤖 {loading ? 'Generating...' : 'AI Auto-Rerank'}</div>
                          <p id="action-auto-desc" className="text-xs text-blue-100 pr-16">
                            The system selects images first, then answers (recommended for beginners)
                          </p>
                        </button>

                        <button
                          onClick={() => submitSelection('direct')}
                          disabled={loading}
                          title="Direct Answer (Not Recommended): use only when retrieval is triggered by system error; no reference images"
                          aria-describedby="action-direct-desc"
                          className="relative w-full text-left p-4 rounded-lg border border-amber-200 bg-amber-50 hover:bg-amber-100 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors"
                        >
                          <span className="absolute top-2 right-2 px-2 py-0.5 text-[10px] font-semibold bg-amber-100 text-amber-700 rounded-full">
                            Not Recommended
                          </span>
                          <div className="text-sm font-semibold text-amber-700 mb-1 pr-24">💬 {loading ? 'Generating...' : 'Direct Answer (Not Recommended)'}</div>
                          <p id="action-direct-desc" className="text-xs text-amber-700">
                            Use only when retrieval is incorrectly triggered by system errors; without reference images, chart style may drift
                          </p>
                        </button>

                        <button
                          onClick={() => submitSelection('manual')}
                          disabled={selectedImages.length === 0 || loading}
                          title="Use Selected Images: answer strictly based on the images you selected (most controllable)"
                          aria-describedby="action-manual-desc"
                          className="w-full text-left p-4 rounded-lg border border-emerald-200 bg-emerald-50 hover:bg-emerald-100 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors"
                        >
                          <div className="text-sm font-semibold text-emerald-700 mb-1">👆 {loading ? 'Generating...' : 'Use Selected Images'}</div>
                          <p id="action-manual-desc" className="text-xs text-emerald-600">
                            Answer strictly based on your selected images (highest controllability)
                          </p>
                        </button>
                      </div>
                    </div>
                  </div>
                )})()}
                
                {/* 显示用户的选择结果 */}
                {message.type === 'selection' && !message.needsSelection && (
                  <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-md">
                    {message.selected_images && message.selected_images.length > 0 ? (
                      <>
                        <div className="text-sm text-green-700 mb-2 font-medium">
                          ✅ {message.selection_mode === 'auto' 
                            ? `AI is reranking the ${message.selected_images.length} selected images` 
                            : `You selected ${message.selected_images.length} images`}
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {message.selected_images.map((path, idx) => (
                            <span key={path} className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded">
                              Image {idx + 1}
                            </span>
                          ))}
                        </div>
                      </>
                    ) : message.selected_images === undefined ? (
                      <div className="text-sm text-green-700 font-medium">✅ Direct answer mode selected</div>
                    ) : (
                      <div className="text-sm text-green-700 font-medium">✅ AI is auto-reranking all candidate images</div>
                    )}
                    <div className="text-xs text-green-600 mt-2">Generating final response...</div>
                  </div>
                )}

                {/* Search到的图表（仅助手消息） */}
                {(message.type === 'assistant' || (message.type === 'selection' && !message.needsSelection)) && message.image_gallery && message.image_gallery.length > 0 && (() => {
                  // 兼容旧格式（string[]）和新格式（ImageCandidate[]）
                  const gallery = message.image_gallery;
                  const isNewFormat = gallery.length > 0 && typeof gallery[0] === 'object' && gallery[0] !== null && 'chart_path' in gallery[0];
                  
                  return (
                  <div className="mt-3">
                    <div className="text-xs text-gray-600 mb-2">📊 Related Charts:</div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
                      {gallery.map((item: string | ImageCandidate, index: number) => {
                        const imagePath = isNewFormat ? (item as ImageCandidate).chart_path : (item as string);
                        const chartType = isNewFormat ? (item as ImageCandidate).chart_type : null;
                        
                        return (
                        <div key={index} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow relative">
                          <img
                            src={`/api/image/${encodeURIComponent(imagePath)}`}
                            alt={`Retrieval Result ${index + 1}`}
                            className="w-full h-64 object-contain bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
                            onClick={() => setEnlargedImage(`/api/image/${encodeURIComponent(imagePath)}`)}
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvueJh+WKoOi9veWksei0pTwvdGV4dD48L3N2Zz4=';
                            }}
                          />
                          {chartType && (
                            <div className="absolute top-2 left-2">
                              <span className="px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded-full font-medium shadow-sm">
                                {chartType}
                              </span>
                            </div>
                          )}
                          <div className="p-3 bg-white">
                            <p className="text-xs text-gray-500 truncate" title={imagePath}>
                              Reference Image {index + 1}
                            </p>
                            <p className="text-xs text-blue-600 mt-1 flex items-center">
                              <span className="mr-1">🔍</span>
                              Click to enlarge
                            </p>
                          </div>
                        </div>
                      )})}
                    </div>
                  </div>
                )})()}
                
                {/* Token 使用量显示 */}
                {(message.type === 'assistant' || message.type === 'selection') && message.token_usage && (
                  <div className="mt-2 pt-2 border-t border-gray-100 flex justify-end">
                    <span className="text-[10px] text-gray-400">
                      Tokens: {message.token_usage.total_tokens.toLocaleString()} 
                      (↑{message.token_usage.prompt_tokens.toLocaleString()} / ↓{message.token_usage.completion_tokens.toLocaleString()})
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {/* 加载指示器 */}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-gray-600 text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 - 仅在对话模式下显示 */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        {/* 模型选择器 */}
        <div className="mb-3">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-600 font-medium">Model:</label>
            <div className="relative">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="text-sm border border-gray-300 rounded-md pl-3 pr-8 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-40"
                style={{ 
                  appearance: 'none',
                  WebkitAppearance: 'none',
                  MozAppearance: 'none'
                }}
              >
                {modelOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
        
        <div className="flex items-end space-x-4">
          {/* 图片上传 */}
          <div className="flex-shrink-0">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="cursor-pointer flex items-center justify-center w-10 h-10 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
              title="Upload image"
            >
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </label>
          </div>

          {/* 文本输入 */}
          <div className="flex-1">
            {userImage && (
              <div className="mb-2 flex items-center space-x-2">
                <img
                  src={draftUserImageUrl || ''}
                  alt="Preview"
                  className="w-12 h-12 object-cover rounded-md"
                />
                <span className="text-sm text-gray-600">{userImage.name}</span>
                <button
                  onClick={() => setUserImage(null)}
                  className="text-red-500 hover:text-red-700"
                >
                  ✕
                </button>
              </div>
            )}
            <div className="flex items-end space-x-2">
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none text-base"
                rows={3}
                style={{ minHeight: '80px', maxHeight: '200px' }}
              />
              <button
                onClick={handleSubmit}
                disabled={loading || !userInput.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
        </div>
        {/* 右侧 Reference Images 面板 */}
        <ReferencePanel
          referencePanelOpen={referencePanelOpen}
          setReferencePanelOpen={setReferencePanelOpen}
          referencePanelWidth={referencePanelWidth}
          setReferencePanelWidth={setReferencePanelWidth}
          currentReferenceImages={currentReferenceImages}
          generatedSvgs={generatedSvgs}
          svgPlaceholderMap={svgPlaceholderMap}
          isResizing={isResizing}
          setIsResizing={setIsResizing}
          setEnlargedImage={setEnlargedImage}
        />
      </div>



      {enlargedImage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setEnlargedImage(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <img
              src={enlargedImage}
              alt="View enlarged"
              className="max-w-full max-h-full object-contain"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={() => setEnlargedImage(null)}
              className="absolute top-4 right-4 w-10 h-10 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full flex items-center justify-center text-white text-xl font-bold transition-colors"
            >
              ✕
            </button>
            <div className="absolute bottom-4 left-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded text-sm">
              Click the blank area to close
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
