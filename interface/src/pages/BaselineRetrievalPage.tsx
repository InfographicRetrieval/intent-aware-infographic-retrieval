import React, { useCallback, useState } from "react";
import * as api from "../api";
import BaselineSearchForm from "../components/baseline/BaselineSearchForm";
import BaselineResultsGrid, { BaselineResult } from "../components/baseline/BaselineResultsGrid";

const BaselineRetrievalPage: React.FC = () => {
  const [query, setQuery] = useState<string>("");
  const [topK, setTopK] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<BaselineResult[]>([]);
  const [error, setError] = useState<string>("");

  const [enlargedImage, setEnlargedImage] = useState<string | null>(null);


  const onSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;

    setLoading(true);
    setError("");
    try {
      const resp = await api.baselineSearch(q, topK);
      setResults(resp?.results || []);
    } catch (e: any) {
      console.error("Baseline search error:", e);
      setError("Search failed. Please try again later.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query, topK]);

  const openImage = useCallback((url: string) => {
    setEnlargedImage(url);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">🔍 Baseline Retrieval</h1>
            <p className="text-sm text-gray-500 mt-1">Enter keywords to view Baseline retrieval results (no login required, no history stored).</p>
          </div>
          <a
            href="#/"
            className="text-sm text-blue-600 hover:text-blue-700 underline underline-offset-4"
            title="Back to main system"
          >
            Main App
          </a>
        </div>

        <BaselineSearchForm
          query={query}
          topK={topK}
          loading={loading}
          onChangeQuery={setQuery}
          onChangeTopK={setTopK}
          onSubmit={onSearch}
        />

        {error && (
          <div className="mt-4 text-red-600 text-sm bg-red-50 border border-red-200 px-3 py-2 rounded">
            {error}
          </div>
        )}

        <div className="mt-6">
          <BaselineResultsGrid results={results} onOpenImage={openImage} />
        </div>

        {!loading && results.length === 0 && !error && (
          <div className="text-center text-gray-400 mt-20">
            <div className="text-6xl mb-4">🔍</div>
            <p>Enter keywords to start searching</p>
          </div>
        )}
      </div>

      {enlargedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setEnlargedImage(null)}
        >
          <div className="relative max-w-5xl max-h-full">
            <img
              src={enlargedImage}
              alt="View enlarged"
              className="max-w-full max-h-full object-contain bg-white rounded"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={() => setEnlargedImage(null)}
              className="absolute top-3 right-3 w-10 h-10 bg-black/40 hover:bg-black/60 rounded-full flex items-center justify-center text-white text-xl font-bold transition-colors"
              title="Close"
            >
              ✕
            </button>
            <div className="absolute bottom-3 left-3 bg-black/50 text-white px-3 py-1 rounded text-sm">
              Click the blank area to close
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BaselineRetrievalPage;
