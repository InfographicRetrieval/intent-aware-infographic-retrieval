import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { ChartTypeNode, RetrievalAspectKey, RetrievalQuerySpec } from './types';
import {
  adjustRetrievalAspectWeight,
  parseRetrievalSpec,
  RETRIEVAL_ASPECT_ORDER,
  updateRetrievalAspect,
} from './utils/retrievalQuery';
import { restoreSvgPlaceholders } from './utils/svgPlaceholders';

const stableCodeId = (language: string, code: string) => {
  const source = `${language}:${code}`;
  let hash = 0;
  for (let i = 0; i < source.length; i += 1) {
    hash = (hash * 31 + source.charCodeAt(i)) >>> 0;
  }
  return `code-${hash.toString(36)}`;
};

// 简易登录组件（美化版 + 内置 Baseline 入口）
export const LoginScreen: React.FC<{ onLogin: (username: string) => void }> = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = username.trim();
    if (!trimmed) {
      setError('Please enter a username');
      return;
    }
    if (trimmed.length > 20) {
      setError('Username cannot exceed 20 characters');
      return;
    }
    if (!/^[a-zA-Z0-9_]+$/.test(trimmed)) {
      setError('Username can only contain letters, numbers, and underscores');
      return;
    }
    onLogin(trimmed);
  };

  return (
    <div className="min-h-screen w-full relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-6">
      {/* 柔和光斑装饰 */}
      <div className="pointer-events-none absolute -top-24 -left-24 w-96 h-96 bg-indigo-500/20 blur-3xl rounded-full" />
      <div className="pointer-events-none absolute -bottom-24 -right-24 w-96 h-96 bg-sky-500/20 blur-3xl rounded-full" />

      <div className="w-full max-w-md">
        <div className="bg-white/10 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/10 p-8">
          <div className="text-center mb-8">
            <div className="text-6xl mb-4">📊</div>
            <h1 className="text-2xl font-bold text-white mb-2">ChartRetriever</h1>
            <p className="text-slate-300">Please enter a username to continue</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-1">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g.: admin, alice, bob"
                className="w-full px-4 py-3 rounded-lg bg-white/10 text-white placeholder:text-slate-400 border border-white/10 focus:outline-none focus:ring-2 focus:ring-sky-400 focus:border-sky-400"
                autoFocus
              />
            </div>

            {error && (
              <div className="text-red-200 text-sm bg-red-500/10 border border-red-500/20 px-3 py-2 rounded">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="w-full py-3 rounded-lg bg-sky-500 text-white hover:bg-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400 focus:ring-offset-2 focus:ring-offset-slate-950 transition-colors font-medium"
            >
              Enter System
            </button>
          </form>

          {/* Demos：不需要登录 */}
          <div className="mt-5 space-y-2">
            <a
              href="#/baseline"
              className="block w-full text-center py-2.5 rounded-lg border border-white/10 text-slate-200 hover:bg-white/10 transition-colors text-sm"
            >
              🔍 Baseline Retrieval Demo (no login)
            </a>

            <a
              href="#/plain-chat"
              className="block w-full text-center py-2.5 rounded-lg border border-white/10 text-slate-200 hover:bg-white/10 transition-colors text-sm"
            >
              💬 Plain Chat Demo (no login)
            </a>
          </div>


          <div className="mt-6 text-center text-xs text-slate-400">
            Each user's conversation history is stored independently
          </div>
        </div>

        <div className="mt-4 text-center text-xs text-slate-500">
          Tip: The Baseline page does not save any history
        </div>
      </div>
    </div>
  );
};

  
  
  
  // 5-Aspect 检索查询表格组件
export const RetrievalQueryTable: React.FC<{
  query: string;
  isSelection?: boolean;
  editable?: boolean;
  editedSpec?: RetrievalQuerySpec | null;
  onEditedSpecChange?: (spec: RetrievalQuerySpec) => void;
  onApplyChanges?: () => void;
  applyDisabled?: boolean;
  isApplying?: boolean;
}> = ({
  query,
  isSelection,
  editable = false,
  editedSpec,
  onEditedSpecChange,
  onApplyChanges,
  applyDisabled = false,
  isApplying = false,
}) => {
  const parsedSpec = parseRetrievalSpec(query);

  const aspectLabels: Record<RetrievalAspectKey, string> = {
    chart_type: '📊 Chart Type',
    content: '📝 Content',
    layout: '📐 Layout',
    style: '🎨 Style',
    illustration: '✨ Decoration',
  };

  const activeSpec = editedSpec || parsedSpec;

  if (!activeSpec) {
    return (
      <>
        <div className={`text-xs mb-2 ${isSelection ? 'text-blue-600' : 'text-gray-600'}`}>🔍 Retrieval Query:</div>
        <div className={`text-sm ${isSelection ? 'text-blue-700' : 'text-gray-700'}`}>{query}</div>
      </>
    );
  }

  const activeAspects = editable
    ? RETRIEVAL_ASPECT_ORDER
    : RETRIEVAL_ASPECT_ORDER.filter((key) => {
        const data = activeSpec[key];
        return data.weight > 0 || data.query.trim();
      });

  if (activeAspects.length === 0) {
    return (
      <>
        <div className={`text-xs mb-2 ${isSelection ? 'text-blue-600' : 'text-gray-600'}`}>🔍 Retrieval Query:</div>
        <div className={`text-sm ${isSelection ? 'text-blue-700' : 'text-gray-700'}`}>No specific retrieval conditions</div>
      </>
    );
  }

  const updateAspectQuery = (key: RetrievalAspectKey, nextQuery: string) => {
    if (!editable || !onEditedSpecChange) return;
    onEditedSpecChange(updateRetrievalAspect(activeSpec, key, { query: nextQuery }));
  };

  const nudgeWeight = (key: RetrievalAspectKey, delta: number) => {
    if (!editable || !onEditedSpecChange) return;
    onEditedSpecChange(adjustRetrievalAspectWeight(activeSpec, key, delta));
  };

  const getWeightBadgeClass = (weight: number) => {
    if (weight >= 0.7) return 'bg-red-100 text-red-700';
    if (weight >= 0.4) return 'bg-yellow-100 text-yellow-700';
    return isSelection ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600';
  };

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className={`text-xs font-medium ${isSelection ? 'text-blue-600' : 'text-gray-600'}`}>
          🔍 Retrieval Spec (5-Facet):
        </div>
        {editable && onApplyChanges && (
          <button
            onClick={onApplyChanges}
            disabled={applyDisabled}
            className="rounded-full bg-blue-600 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-200"
          >
            {isApplying ? 'Re-retrieving...' : 'Apply & Re-retrieve'}
          </button>
        )}
      </div>
      <div className="overflow-x-auto rounded-lg border border-blue-100 bg-white/60">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className={`${isSelection ? 'bg-blue-100' : 'bg-gray-100'}`}>
              <th className={`text-left py-2 px-3 text-xs font-semibold ${isSelection ? 'text-blue-700' : 'text-gray-600'} border-b ${isSelection ? 'border-blue-200' : 'border-gray-200'}`} style={{ width: '120px' }}>
                Facet
              </th>
              <th className={`text-left py-2 px-3 text-xs font-semibold ${isSelection ? 'text-blue-700' : 'text-gray-600'} border-b ${isSelection ? 'border-blue-200' : 'border-gray-200'}`}>
                Query Content
              </th>
              <th className={`text-center py-2 px-3 text-xs font-semibold ${isSelection ? 'text-blue-700' : 'text-gray-600'} border-b ${isSelection ? 'border-blue-200' : 'border-gray-200'}`} style={{ width: editable ? '170px' : '90px' }}>
                Weight
              </th>
            </tr>
          </thead>
          <tbody>
            {activeAspects.map((key) => {
              const data = activeSpec[key];
              return (
                <tr key={key} className={`${isSelection ? 'hover:bg-blue-100/40' : 'hover:bg-gray-50'}`}>
                  <td className={`py-2 px-3 text-xs font-medium ${isSelection ? 'text-blue-700' : 'text-gray-600'} border-b ${isSelection ? 'border-blue-100' : 'border-gray-100'} align-top`}>
                    {aspectLabels[key]}
                  </td>
                  <td className={`py-2 px-3 text-xs ${isSelection ? 'text-gray-700' : 'text-gray-700'} border-b ${isSelection ? 'border-blue-100' : 'border-gray-100'}`}>
                    {editable ? (
                      <textarea
                        value={data.query}
                        onChange={(e) => updateAspectQuery(key, e.target.value)}
                        rows={2}
                        className="w-full resize-none rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200"
                      />
                    ) : data.query ? (
                      data.query
                    ) : (
                      <span className="text-gray-400 italic">-</span>
                    )}
                  </td>
                  <td className={`py-2 px-3 text-xs text-center border-b ${isSelection ? 'border-blue-100' : 'border-gray-100'}`}>
                    {editable ? (
                      <div className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-2 py-1">
                        <button
                          type="button"
                          onClick={() => nudgeWeight(key, -0.1)}
                          className="flex h-6 w-6 items-center justify-center rounded-full bg-white text-rose-500 shadow-sm transition-colors hover:bg-rose-100"
                        >
                          -
                        </button>
                        <span className={`min-w-8 font-semibold ${getWeightBadgeClass(data.weight).split(' ')[1]}`}>
                          {data.weight.toFixed(1)}
                        </span>
                        <button
                          type="button"
                          onClick={() => nudgeWeight(key, 0.1)}
                          className="flex h-6 w-6 items-center justify-center rounded-full bg-white text-rose-500 shadow-sm transition-colors hover:bg-rose-100"
                        >
                          +
                        </button>
                      </div>
                    ) : (
                      <span className={`inline-flex items-center justify-center px-2 py-0.5 rounded-full text-xs font-medium ${getWeightBadgeClass(data.weight)}`}>
                        {data.weight.toFixed(1)}
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
  
  // Chart Type 树形节点组件
export const ChartTypeTreeNode: React.FC<{
    node: ChartTypeNode;
    selectedNodes: Set<string>;
    expandedNodes: Set<string>;
    onToggleSelect: (nodeName: string) => void;
    onToggleExpand: (nodeName: string) => void;
    level: number;
  }> = ({ node, selectedNodes, expandedNodes, onToggleSelect, onToggleExpand, level }) => {
    const isSelected = selectedNodes.has(node.name);
    const isExpanded = expandedNodes.has(node.name);
    const hasChildren = node.children.length > 0;
    const indentStyle = { paddingLeft: `${level * 16}px` };
  
    return (
      <div>
        <div 
          className={`flex items-center py-1 px-1 hover:bg-blue-50 cursor-pointer rounded ${
            isSelected ? 'bg-blue-50' : ''
          }`}
          style={indentStyle}
        >
          {/* 展开/折叠图标 */}
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleExpand(node.name);
              }}
              className="w-4 h-4 flex items-center justify-center mr-1 hover:bg-gray-200 rounded"
            >
              <span className="text-xs text-gray-600">
                {isExpanded ? '▼' : '▶'}
              </span>
            </button>
          )}
          {!hasChildren && <span className="w-4 mr-1" />}
          
          {/* 选择框 */}
          <input
            type="checkbox"
            checked={isSelected}
            onChange={() => onToggleSelect(node.name)}
            className="mr-2 cursor-pointer"
            onClick={(e) => e.stopPropagation()}
          />
          
          {/* 节点信息 */}
          <div 
            className="flex-1 flex items-center justify-between text-xs"
            onClick={() => onToggleSelect(node.name)}
          >
            <span className={`${isSelected ? 'font-medium text-blue-700' : 'text-gray-700'}`}>
              {node.name}
            </span>
            <span className="text-xs text-gray-500 ml-2">
              {node.count.toLocaleString()}
            </span>
          </div>
        </div>
        
        {/* 子节点 */}
        {hasChildren && isExpanded && (
          <div>
            {node.children.map((child) => (
              <ChartTypeTreeNode
                key={child.name}
                node={child}
                selectedNodes={selectedNodes}
                expandedNodes={expandedNodes}
                onToggleSelect={onToggleSelect}
                onToggleExpand={onToggleExpand}
                level={level + 1}
              />
            ))}
          </div>
        )}
      </div>
    );
  };


// Markdown渲染组件
export const MarkdownRenderer: React.FC<{ content: string; svgPlaceholderMap?: Record<string, string> }> = ({
  content,
  svgPlaceholderMap = {},
}) => {
  const [copiedCode, setCopiedCode] = React.useState<string | null>(null);
  const [showSvgPreview, setShowSvgPreview] = React.useState<{ [key: string]: boolean }>({});

  const copyToClipboard = (code: string, id: string) => {
    // 检查是否在安全环境中，优先使用现代 Clipboard API
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(code).then(() => {
        setCopiedCode(id);
        setTimeout(() => setCopiedCode(null), 2000);
      });
    } else {
      // 在不安全环境 (http) 或不支持的环境下，使用备用方案
      const textArea = document.createElement('textarea');
      textArea.value = code;
      textArea.style.position = 'fixed';
      textArea.style.left = '-9999px';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopiedCode(id);
        setTimeout(() => setCopiedCode(null), 2000);
      } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
      }
      document.body.removeChild(textArea);
    }
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // 自定义代码块样式
        code: ({ node, className, children, ...props }: any) => {
          const match = /language-(\w+)/.exec(className || '');
          const inline = !match;
          const language = match ? match[1] : '';
          const codeString = String(children).replace(/\n$/, '');
          const codeId = stableCodeId(language, codeString);
          
          if (!inline && language === 'svg') {
            const restoredSvgCode = showSvgPreview[codeId]
              ? restoreSvgPlaceholders(codeString, svgPlaceholderMap)
              : codeString;
            // SVG 代码块特殊处理
            return (
              <div className="my-4 border border-gray-300 rounded-lg overflow-hidden bg-white shadow-sm">
                {/* SVG 预览/代码切换按钮 */}
                <div className="bg-gray-100 px-3 py-2 flex items-center justify-between border-b border-gray-300">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold text-gray-700 uppercase">SVG Code</span>
                    <button
                      onClick={() => setShowSvgPreview(prev => ({ ...prev, [codeId]: !prev[codeId] }))}
                      className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                    >
                      {showSvgPreview[codeId] ? 'Show Code' : 'Preview SVG'}
                    </button>
                  </div>
                  <button
                    onClick={() => copyToClipboard(restoreSvgPlaceholders(codeString, svgPlaceholderMap), codeId)}
                    className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors flex items-center space-x-1"
                  >
                    {copiedCode === codeId ? (
                      <>
                        <span>✓</span>
                        <span>Copied</span>
                      </>
                    ) : (
                      <>
                        <span>📋</span>
                        <span>Copy code</span>
                      </>
                    )}
                  </button>
                </div>
                
                {/* SVG 预览或代码显示 */}
                {showSvgPreview[codeId] ? (
                  <div className="p-4 bg-gray-50 flex items-center justify-center min-h-[200px]">
                    <div 
                      dangerouslySetInnerHTML={{ __html: restoredSvgCode }}
                      className="max-w-full"
                    />
                  </div>
                ) : (
                  <pre className="bg-gray-800 text-gray-100 p-3 overflow-x-auto text-sm m-0">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                )}
              </div>
            );
          }
          
          // 其他代码块
          return !inline ? (
            <div className="relative my-2">
              <button
                onClick={() => copyToClipboard(codeString, codeId)}
                className="absolute top-2 right-2 text-xs px-2 py-1 bg-gray-700 text-gray-200 rounded hover:bg-gray-600 transition-colors"
              >
                {copiedCode === codeId ? '✓ Copied' : '📋 Copy'}
              </button>
              <pre className="bg-gray-800 text-gray-100 rounded-md p-3 overflow-x-auto text-sm">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            </div>
          ) : (
            <code className="bg-gray-200 text-gray-800 px-1 py-0.5 rounded text-sm font-mono" {...props}>
              {children}
            </code>
          );
        },
        // 自定义表格样式
        table: ({ children }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full border border-gray-300 rounded-lg bg-white shadow-sm">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-gray-300 px-3 py-2 bg-gray-100 font-semibold text-left text-sm">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-gray-300 px-3 py-2 text-sm">
            {children}
          </td>
        ),
        // 自定义列表样式
        ul: ({ children }) => (
          <ul className="list-disc list-inside my-2 space-y-1">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-inside my-2 space-y-1">
            {children}
          </ol>
        ),
        // 自定义标题样式
        h1: ({ children }) => (
          <h1 className="text-2xl font-bold my-4 text-gray-900">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-xl font-bold my-3 text-gray-900">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-lg font-semibold my-2 text-gray-900">
            {children}
          </h3>
        ),
        // 自定义引用样式
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-500 pl-4 my-3 italic text-gray-700 bg-blue-50 py-2 rounded-r-md">
            {children}
          </blockquote>
        ),
        // 自定义链接样式
        a: ({ children, href }) => (
          <a 
            href={href} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-800 underline"
          >
            {children}
          </a>
        ),
        // 自定义段落样式
        p: ({ children }) => (
          <p className="my-2 leading-relaxed">
            {children}
          </p>
        ),
        // 自定义强调样式
        strong: ({ children }) => (
          <strong className="font-bold text-gray-900">
            {children}
          </strong>
        ),
        em: ({ children }) => (
          <em className="italic text-gray-800">
            {children}
          </em>
        ),
        // 自定义分隔线样式
        hr: () => (
          <hr className="my-4 border-gray-300" />
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
};
