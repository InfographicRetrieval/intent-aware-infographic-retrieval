import React, { useEffect, useMemo, useRef, useState } from 'react';
import { restoreSvgPlaceholders } from '../utils/svgPlaceholders';

type GeneratedSvgItem = {
  key: string;
  messageId: string;
  timestamp: Date;
  svgCode: string;
  pngUrl?: string;
};

interface ReferencePanelProps {
  referencePanelOpen: boolean;
  setReferencePanelOpen: (open: boolean) => void;
  referencePanelWidth: number;
  setReferencePanelWidth: (width: number) => void;
  currentReferenceImages: string[];
  generatedSvgs: GeneratedSvgItem[];
  svgPlaceholderMap: Record<string, string>;
  isResizing: boolean;
  setIsResizing: (resizing: boolean) => void;
  setEnlargedImage: (url: string | null) => void;
}

type RenderStatus = 'idle' | 'queued' | 'rendering' | 'success' | 'error' | 'timeout' | 'unsupported';

type RenderState = {
  status: RenderStatus;
  pngUrl?: string; // local generated blob url
  error?: string;
  startedAt?: number;
  finishedAt?: number;
  attempts: number;
};

const DEFAULT_TIMEOUT_MS = 6000;
const DEFAULT_CONCURRENCY = 2;

async function rasterizeSvgInMainThread(svgText: string): Promise<Blob> {
  const svgBlob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' });
  const svgUrl = URL.createObjectURL(svgBlob);

  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load SVG into <img>.'));
      img.src = svgUrl;
    });

    const width = Math.max(1, image.naturalWidth || image.width || 1);
    const height = Math.max(1, image.naturalHeight || image.height || 1);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to create canvas 2D context.');
    }

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);

    const pngBlob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Canvas export returned an empty blob.'));
        }
      }, 'image/png');
    });

    return pngBlob;
  } finally {
    try { URL.revokeObjectURL(svgUrl); } catch {}
  }
}

/**
 * Create a dedicated worker (from Blob URL) that rasterizes SVG string into PNG (Blob).
 * Uses OffscreenCanvas + createImageBitmap when available.
 *
 * This isolates "bad SVG" / heavy parsing from the main thread.
 * If it hangs, we can terminate the worker without affecting subsequent tasks.
 */
function createSvgRasterWorker(): Worker {
  const workerCode = `
    self.onmessage = async (e) => {
      const { id, svgText, timeoutMs } = e.data || {};
      const reply = (payload) => self.postMessage({ id, ...payload });

      try {
        if (!svgText || typeof svgText !== 'string') {
          reply({ ok: false, error: 'Empty svgText.' });
          return;
        }

        // OffscreenCanvas is required for reliable worker-side rasterization.
        if (typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap === 'undefined') {
          reply({ ok: false, unsupported: true, error: 'OffscreenCanvas/createImageBitmap not available in this browser.' });
          return;
        }

        // Make sure it is a valid svg root at least minimally.
        // Avoid script/foreignObject; remove on* handlers and javascript: urls.
        const sanitize = (raw) => {
          // lightweight sanitize without DOMParser (DOMParser not available in worker consistently)
          // strip <script> and <foreignObject>
          let s = raw.replace(/<script[\\s\\S]*?<\\/script>/gi, '');
          s = s.replace(/<foreignObject[\\s\\S]*?<\\/foreignObject>/gi, '');
          // strip inline event handlers: onload=, onclick= ...
          s = s.replace(/\\son\\w+\\s*=\\s*(['"]).*?\\1/gi, '');
          // strip javascript: hrefs
          s = s.replace(/(href|xlink:href)\\s*=\\s*(['"])javascript:[\\s\\S]*?\\2/gi, '$1=$2$2');
          return s;
        };

        const cleaned = sanitize(svgText);

        const svgBlob = new Blob([cleaned], { type: 'image/svg+xml;charset=utf-8' });

        // Load SVG as bitmap
        const bitmap = await createImageBitmap(svgBlob);

        // Decide canvas size (bitmap width/height inferred)
        const w = Math.max(1, bitmap.width || 1);
        const h = Math.max(1, bitmap.height || 1);

        const canvas = new OffscreenCanvas(w, h);
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reply({ ok: false, error: 'Failed to get 2D context.' });
          try { bitmap.close(); } catch {}
          return;
        }

        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(bitmap, 0, 0, w, h);

        try { bitmap.close(); } catch {}

        // Convert to PNG blob
        const pngBlob = await canvas.convertToBlob({ type: 'image/png' });

        // Transfer as ArrayBuffer to main thread
        const buf = await pngBlob.arrayBuffer();
        reply({ ok: true, pngArrayBuffer: buf }, [buf]);
      } catch (err) {
        reply({ ok: false, error: (err && err.message) ? err.message : String(err) });
      }
    };
  `;
  const blobUrl = URL.createObjectURL(new Blob([workerCode], { type: 'text/javascript' }));
  const w = new Worker(blobUrl);
  // revoke after worker created; the worker holds its own reference
  URL.revokeObjectURL(blobUrl);
  return w;
}

/**
 * Rasterize SVG string into a Blob URL (png) with:
 * - dedicated worker (terminated on finish/timeout)
 * - hard timeout
 */
async function rasterizeSvgToPngUrl(svgText: string, timeoutMs: number): Promise<{ status: RenderStatus; pngUrl?: string; error?: string }> {
  const worker = createSvgRasterWorker();
  const id = `svg-render-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  let timeoutHandle: any = null;

  try {
    const result = await new Promise<{ ok: boolean; pngArrayBuffer?: ArrayBuffer; error?: string; unsupported?: boolean }>((resolve, reject) => {
      timeoutHandle = setTimeout(() => {
        reject(new Error('TIMEOUT'));
      }, timeoutMs);

      worker.onmessage = (e: MessageEvent) => {
        const data = e.data || {};
        if (data.id !== id) return;
        resolve(data);
      };
      worker.onerror = (e) => {
        reject(new Error(e?.message || 'Worker error'));
      };

      worker.postMessage({ id, svgText, timeoutMs });
    });

    if (!result.ok) {
      if (result.unsupported) {
        return { status: 'unsupported', error: result.error || 'Unsupported browser.' };
      }
      return { status: 'error', error: result.error || 'Unknown error.' };
    }

    if (!result.pngArrayBuffer) {
      return { status: 'error', error: 'Missing PNG buffer.' };
    }

    const pngBlob = new Blob([result.pngArrayBuffer], { type: 'image/png' });
    const url = URL.createObjectURL(pngBlob);
    return { status: 'success', pngUrl: url };
  } catch (e: any) {
    if (String(e?.message || e) === 'TIMEOUT') {
      return { status: 'timeout', error: `Render timeout after ${timeoutMs}ms` };
    }
    return { status: 'error', error: e?.message ? e.message : String(e) };
  } finally {
    if (timeoutHandle) clearTimeout(timeoutHandle);
    // If it hangs, terminating ensures this render never blocks next tasks.
    worker.terminate();
  }
}

const ReferencePanel: React.FC<ReferencePanelProps> = ({
  referencePanelOpen,
  setReferencePanelOpen,
  referencePanelWidth,
  setReferencePanelWidth,
  currentReferenceImages,
  generatedSvgs,
  svgPlaceholderMap,
  isResizing,
  setIsResizing,
  setEnlargedImage,
}) => {
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);
  const [svgPreviewUrls, setSvgPreviewUrls] = useState<Record<string, string>>({});
  const svgEndRef = useRef<HTMLDivElement>(null);

  // Local per-item render states (only for items whose pngUrl is not provided)
  const [renderStates, setRenderStates] = useState<Record<string, RenderState>>({});

  // A small queue to process rendering tasks without blocking each other
  const queueRef = useRef<string[]>([]);
  const runningRef = useRef<number>(0);
  const mountedRef = useRef<boolean>(false);

  const hasAnyContent = currentReferenceImages.length > 0 || generatedSvgs.length > 0;

  // 处理拖动调整面板宽度
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  const handleCopySvg = async (item: GeneratedSvgItem) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(item.svgCode);
      } else {
        const textArea = document.createElement('textarea');
        textArea.value = item.svgCode;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
      }
      setActionFeedback(`${item.key}:svg`);
      setTimeout(() => setActionFeedback(null), 1500);
    } catch (e) {
      console.error('Failed to copy SVG code:', e);
    }
  };

  const sortedSvgs = useMemo(() => {
    const arr = generatedSvgs.map((item) => ({
      ...item,
      svgCode: restoreSvgPlaceholders(item.svgCode, svgPlaceholderMap),
    }));
    arr.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
    return arr;
  }, [generatedSvgs, svgPlaceholderMap]);

  // Auto scroll to newest
  useEffect(() => {
    if (!referencePanelOpen) return;
    if (generatedSvgs.length === 0) return;
    svgEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [referencePanelOpen, generatedSvgs.length]);

  // Mark mounted
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      setSvgPreviewUrls((prev) => {
        Object.values(prev).forEach((url) => {
          try { URL.revokeObjectURL(url); } catch {}
        });
        return {};
      });
      // cleanup blob urls we created
      setRenderStates((prev) => {
        Object.values(prev).forEach((st) => {
          if (st.pngUrl && st.status === 'success') {
            try {
              URL.revokeObjectURL(st.pngUrl);
            } catch {}
          }
        });
        return {};
      });
    };
  }, []);

  useEffect(() => {
    const nextPreviewUrls: Record<string, string> = {};
    sortedSvgs.forEach((item) => {
      nextPreviewUrls[item.key] = URL.createObjectURL(
        new Blob([item.svgCode], { type: 'image/svg+xml;charset=utf-8' })
      );
    });

    setSvgPreviewUrls(nextPreviewUrls);

    return () => {
      Object.values(nextPreviewUrls).forEach((url) => {
        try { URL.revokeObjectURL(url); } catch {}
      });
    };
  }, [sortedSvgs]);

  // Ensure renderStates has entries for items missing pngUrl; enqueue them
  useEffect(() => {
    const needRender: string[] = [];
    const next: Record<string, RenderState> = {};

    for (const item of sortedSvgs) {
      const key = item.key;
      const existing = renderStates[key];

      // If backend already provides pngUrl, we don't need local render
      if (item.pngUrl) continue;

      if (!existing) {
        next[key] = { status: 'queued', attempts: 0 };
        needRender.push(key);
      } else {
        // keep existing state
      }
    }

    if (Object.keys(next).length > 0) {
      setRenderStates((prev) => ({ ...prev, ...next }));
    }

    // push to queue (dedupe)
    if (needRender.length > 0) {
      const q = queueRef.current;
      for (const k of needRender) {
        if (!q.includes(k)) q.push(k);
      }
      // kick runner
      void pumpQueue();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortedSvgs]);

  // Queue runner
  const pumpQueue = async () => {
    // avoid too many loops
    while (runningRef.current < DEFAULT_CONCURRENCY && queueRef.current.length > 0) {
      const key = queueRef.current.shift();
      if (!key) break;

      const item = sortedSvgs.find((x) => x.key === key);
      if (!item) continue;

      // If pngUrl appears later, skip
      if (item.pngUrl) continue;

      // If already succeeded, skip
      const st = renderStates[key];
      if (st?.status === 'success') continue;

      runningRef.current += 1;

      // fire-and-forget but accounted by runningRef
      (async () => {
        try {
          setRenderStates((prev) => {
            const prevSt = prev[key] || { status: 'idle', attempts: 0 };
            return {
              ...prev,
              [key]: {
                ...prevSt,
                status: 'rendering',
                attempts: (prevSt.attempts || 0) + 1,
                startedAt: Date.now(),
                error: undefined,
              },
            };
          });

          const { status, pngUrl, error } = await rasterizeSvgToPngUrl(item.svgCode, DEFAULT_TIMEOUT_MS);

          if (!mountedRef.current) return;

          setRenderStates((prev) => {
            // revoke old pngUrl if any
            const old = prev[key];
            if (old?.pngUrl && old.pngUrl !== pngUrl) {
              try { URL.revokeObjectURL(old.pngUrl); } catch {}
            }
            return {
              ...prev,
              [key]: {
                ...(prev[key] || { attempts: 0 }),
                status,
                pngUrl: pngUrl || prev[key]?.pngUrl,
                error,
                finishedAt: Date.now(),
              },
            };
          });
        } finally {
          runningRef.current -= 1;
          // continue pumping
          if (mountedRef.current) {
            void pumpQueue();
          }
        }
      })();
    }
  };

  const retryRender = (key: string) => {
    // enqueue again
    const q = queueRef.current;
    if (!q.includes(key)) q.push(key);
    setRenderStates((prev) => ({
      ...prev,
      [key]: { ...(prev[key] || { attempts: 0 }), status: 'queued', error: undefined },
    }));
    void pumpQueue();
  };

  const openRawSvgInNewTab = (svgText: string) => {
    try {
      const restoredSvgText = restoreSvgPlaceholders(svgText, svgPlaceholderMap);
      const blob = new Blob([restoredSvgText], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank', 'noopener,noreferrer');
      // revoke later
      setTimeout(() => {
        try { URL.revokeObjectURL(url); } catch {}
      }, 10_000);
    } catch (e) {
      console.error('Failed to open raw SVG:', e);
    }
  };

  const downloadSvg = (svgText: string, version: number) => {
    try {
      const restoredSvgText = restoreSvgPlaceholders(svgText, svgPlaceholderMap);
      const blob = new Blob([restoredSvgText], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `generated-chart-v${version}.svg`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      setTimeout(() => {
        try { URL.revokeObjectURL(url); } catch {}
      }, 10_000);
    } catch (e) {
      console.error('Failed to download restored SVG:', e);
    }
  };

  const triggerDownload = (url: string, filename: string) => {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  };

  const downloadPng = async (item: GeneratedSvgItem, version: number, pngUrl?: string) => {
    try {
      if (pngUrl) {
        triggerDownload(pngUrl, `generated-chart-v${version}.png`);
        return;
      }

      const pngBlob = await rasterizeSvgInMainThread(item.svgCode);
      const url = URL.createObjectURL(pngBlob);
      triggerDownload(url, `generated-chart-v${version}.png`);
      setTimeout(() => {
        try { URL.revokeObjectURL(url); } catch {}
      }, 10_000);
    } catch (e) {
      console.error('Failed to download PNG:', e);
      alert('Failed to export PNG from this SVG.');
    }
  };

  const copyPng = async (item: GeneratedSvgItem, key: string, pngUrl?: string) => {
    try {
      const blob = pngUrl
        ? await (await fetch(pngUrl)).blob()
        : await rasterizeSvgInMainThread(item.svgCode);
      if (navigator.clipboard?.write && typeof ClipboardItem !== 'undefined') {
        await navigator.clipboard.write([
          new ClipboardItem({
            [blob.type || 'image/png']: blob,
          }),
        ]);
        setActionFeedback(`${key}:png`);
        setTimeout(() => setActionFeedback(null), 1500);
        return;
      }
      throw new Error('Clipboard image write is not supported in this browser.');
    } catch (e) {
      console.error('Failed to copy PNG:', e);
      alert('Copy PNG is not supported in this browser. You can still download it.');
    }
  };

  const secondaryButtonClass =
    'inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:border-slate-100 disabled:bg-slate-50 disabled:text-slate-400';
  const primaryButtonClass =
    'inline-flex items-center justify-center rounded-xl bg-blue-600 px-3 py-2 text-xs font-medium text-white shadow-sm transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-200';

  const renderStatusText = (st?: RenderState) => {
    if (!st) return 'Waiting for render...';
    switch (st.status) {
      case 'queued':
        return 'Waiting in render queue...';
      case 'rendering':
        return 'Rendering SVG → PNG ...';
      case 'timeout':
        return `Render timeout (${DEFAULT_TIMEOUT_MS}ms)`;
      case 'unsupported':
        return 'Browser does not support Worker rendering (retry or view original SVG)';
      case 'error':
        return 'Render failed (retry or view original SVG)';
      case 'success':
        return 'Render complete';
      default:
        return 'Waiting for render...';
    }
  };

  return (
    <>
      {/* Reference Images 面板 */}
      <div
        className={`${referencePanelOpen && hasAnyContent ? '' : 'w-0'} ${isResizing ? '' : 'transition-all duration-300'} overflow-hidden bg-white border-l border-gray-200 flex flex-col relative shrink-0`}
        style={{
          width: referencePanelOpen && hasAnyContent ? `${referencePanelWidth}px` : '0px'
        }}
      >
        {/* 拖动手柄 */}
        {referencePanelOpen && hasAnyContent && (
          <div
            className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500 transition-colors z-20"
            onMouseDown={handleMouseDown}
            style={{
              background: isResizing ? '#3b82f6' : 'transparent'
            }}
          >
            <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-1 h-12 bg-gray-300 rounded-r" />
          </div>
        )}

        {hasAnyContent && (
          <>
            {/* 面板头部 */}
            <div className="p-3 border-b border-gray-200 bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">📊</span>
                  <h3 className="text-sm font-semibold text-gray-900">Reference Images</h3>
                </div>
                <button
                  onClick={() => setReferencePanelOpen(false)}
                  className="p-1 hover:bg-gray-200 rounded transition-colors"
                  title="Close panel"
                >
                  <svg className="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {currentReferenceImages.length} reference images · {sortedSvgs.length} SVG versions
              </p>
            </div>

            {/* 图片列表和SVG的统一可滚动容器 */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {/* 图片列表 */}
              {currentReferenceImages.map((imagePath, index) => (
                <div key={index} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow bg-white">
                  <div className="relative w-full">
                    <img
                      src={`/api/image/${encodeURIComponent(imagePath)}`}
                      alt={`Reference ${index + 1}`}
                      className="w-full cursor-pointer"
                      style={{
                        height: 'auto',
                        objectFit: 'contain',
                        backgroundColor: '#f9fafb'
                      }}
                      onClick={() => setEnlargedImage(`/api/image/${encodeURIComponent(imagePath)}`)}
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src =
                          'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuWbvueJh+WKoOi9veWksei0pTwvdGV4dD48L3N2Zz4=';
                      }}
                    />
                    <div className="absolute top-2 left-2 bg-blue-600 text-white text-xs px-2 py-1 rounded-full font-medium shadow-md">
                      #{index + 1}
                    </div>
                  </div>
                  <div className="px-2 py-1.5 bg-gray-50 border-t border-gray-100">
                    <p className="text-xs text-gray-700 font-medium">Reference Image {index + 1}</p>
                    <p className="text-xs text-blue-600 mt-0.5 flex items-center">
                      <span className="mr-1">🔍</span>
                      Click to enlarge
                    </p>
                  </div>
                </div>
              ))}

              {/* SVG 版本列表（全部展开，旧->新，默认滚到最新） */}
              {sortedSvgs.map((item, idx) => {
                const local = renderStates[item.key];
                const effectivePngUrl = item.pngUrl || local?.pngUrl;
                const svgPreviewUrl = svgPreviewUrls[item.key];
                const effectivePreviewUrl = effectivePngUrl || svgPreviewUrl;

                return (
                  <div key={item.key} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow bg-white">
                    {/* 头部 */}
                    <div className="p-2 border-b border-gray-200 bg-gray-100">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <span className="text-sm">🎨</span>
                          <h4 className="text-xs font-semibold text-gray-900">Generated SVG · v{idx + 1}</h4>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="grid grid-cols-2 gap-2">
                            <button
                              onClick={() => downloadSvg(item.svgCode, idx + 1)}
                              className={primaryButtonClass}
                              title="Download SVG with placeholders restored"
                            >
                              Download SVG
                            </button>
                            <button
                              onClick={() => handleCopySvg(item)}
                              className={secondaryButtonClass}
                              title="Copy SVG source"
                            >
                              {actionFeedback === `${item.key}:svg` ? 'Copied SVG' : 'Copy SVG'}
                            </button>
                            <button
                              onClick={() => downloadPng(item, idx + 1, effectivePngUrl)}
                              disabled={!effectivePngUrl && !svgPreviewUrl}
                              className={secondaryButtonClass}
                              title={effectivePngUrl || svgPreviewUrl ? 'Download PNG image' : 'PNG preview is not available yet'}
                            >
                              Download PNG
                            </button>
                            <button
                              onClick={() => copyPng(item, item.key, effectivePngUrl)}
                              disabled={!effectivePngUrl && !svgPreviewUrl}
                              className={secondaryButtonClass}
                              title={effectivePngUrl || svgPreviewUrl ? 'Copy PNG image' : 'PNG preview is not available yet'}
                            >
                              {actionFeedback === `${item.key}:png` ? 'Copied PNG' : 'Copy PNG'}
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 图片预览 */}
                    <div className="relative w-full">
                      {effectivePreviewUrl ? (
                        <img
                          src={effectivePreviewUrl}
                          alt={`Generated v${idx + 1}`}
                          className="w-full cursor-pointer"
                          style={{
                            height: 'auto',
                            objectFit: 'contain',
                            backgroundColor: '#ffffff'
                          }}
                          onClick={() => setEnlargedImage(effectivePreviewUrl || null)}
                        />
                      ) : (
                        <div className="w-full py-8 px-3 flex flex-col items-center justify-center text-xs text-gray-600 bg-white space-y-2">
                          <div className="text-gray-500">{renderStatusText(local)}</div>

                          {/* Show error detail */}
                          {(local?.status === 'error' || local?.status === 'timeout' || local?.status === 'unsupported') && (
                            <div className="max-w-full text-[11px] text-red-600 break-words">
                              {local?.error || 'Unknown error'}
                            </div>
                          )}

                          {/* Minimal controls even while rendering */}
                          <div className="flex items-center space-x-2">
                            {!item.pngUrl && (local?.status === 'error' || local?.status === 'timeout' || local?.status === 'unsupported') && (
                              <>
                                <button
                                  onClick={() => retryRender(item.key)}
                                  className="text-xs px-2 py-1 bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200 transition-colors"
                                  title="Retry render"
                                >
                                  Retry PNG Render
                                </button>
                                <button
                                  onClick={() => openRawSvgInNewTab(item.svgCode)}
                                  className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                                  title="Open original SVG in new tab"
                                >
                                  Open Original SVG
                                </button>
                              </>
                            )}
                            {!item.pngUrl && (local?.status === 'rendering' || local?.status === 'queued' || !local) && (
                              <button
                                onClick={() => openRawSvgInNewTab(item.svgCode)}
                                className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                                title="Open original SVG"
                              >
                                Open Original SVG
                              </button>
                            )}
                            {!item.pngUrl && (local?.status === 'rendering' || local?.status === 'queued') && (
                              <span className="text-[11px] text-gray-400">
                                Concurrency {DEFAULT_CONCURRENCY} · Timeout {DEFAULT_TIMEOUT_MS}ms
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="px-2 py-1.5 bg-gray-50 border-t border-gray-100">
                      <p className="text-xs text-gray-700 font-medium">Generated Chart v{idx + 1}</p>
                      <div className="mt-0.5 flex items-center justify-between gap-2">
                        <p className="text-xs text-blue-600 flex items-center">
                          <span className="mr-1">🔍</span>
                          Click to enlarge
                        </p>
                        {!effectivePngUrl && svgPreviewUrl && (
                          <span className="text-[11px] text-amber-600">
                            Previewing original SVG
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}

              <div ref={svgEndRef} />
            </div>
          </>
        )}
      </div>

      {/* 右侧面板折叠按钮（当面板关闭且有内容时显示） */}
      {!referencePanelOpen && hasAnyContent && (
        <button
          onClick={() => setReferencePanelOpen(true)}
          className="fixed right-0 top-1/2 transform -translate-y-1/2 bg-blue-600 text-white p-2 rounded-l-lg shadow-lg hover:bg-blue-700 transition-colors z-10"
          title="Show reference images"
        >
          <div className="flex flex-col items-center">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="text-xs mt-1">📊</span>
          </div>
        </button>
      )}
    </>
  );
};

export default ReferencePanel;
