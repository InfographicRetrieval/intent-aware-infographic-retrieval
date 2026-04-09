import React from "react";

type Props = {
  query: string;
  topK: number;
  loading: boolean;
  onChangeQuery: (v: string) => void;
  onChangeTopK: (v: number) => void;
  onSubmit: () => void;
};

const BaselineSearchForm: React.FC<Props> = ({ query, topK, loading, onChangeQuery, onChangeTopK, onSubmit }) => {
  const disabled = loading || query.trim().length === 0;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <div className="flex flex-col md:flex-row md:items-center gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => onChangeQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              if (!disabled) onSubmit();
            }
          }}
          placeholder="Enter search query..."
          className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
        />

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">TopK</span>
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => onChangeTopK(Math.max(1, Math.min(50, Number(e.target.value) || 10)))}
            className="w-24 px-3 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
          />
        </div>

        <button
          onClick={onSubmit}
          disabled={disabled}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </div>
  );
};

export default BaselineSearchForm;
