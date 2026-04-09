import React from "react";

export type BaselineResult = {
  rank: number;
  chart_path: string;
  chart_type: string;
  similarity_score: number;
};

type Props = {
  results: BaselineResult[];
  onOpenImage: (imageUrl: string) => void;
};

const BaselineResultsGrid: React.FC<Props> = ({ results, onOpenImage }) => {
  if (!results || results.length === 0) return null;

  return (
    <div>
      <div className="text-sm text-gray-500 mb-4">Returned {results.length} results:</div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {results.map((result) => {
          const imageUrl = `/api/image/${encodeURIComponent(result.chart_path)}`;
          return (
            <div
              key={`${result.rank}-${result.chart_path}`}
              className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow bg-white"
            >
              <div className="relative">
                <button
                  type="button"
                  className="block w-full"
                  onClick={() => onOpenImage(imageUrl)}
                  title="Click to enlarge"
                >
                  <img
                    src={imageUrl}
                    alt={`Result ${result.rank}`}
                    className="w-full h-40 object-contain bg-gray-50 hover:bg-gray-100"
                    loading="lazy"
                  />
                </button>

                <div className="absolute top-2 left-2 bg-blue-600 text-white text-xs px-2 py-1 rounded-full font-bold">
                  #{result.rank}
                </div>
                <div className="absolute top-2 right-2 bg-gray-800 text-white text-xs px-2 py-1 rounded-full">
                  {Number(result.similarity_score).toFixed(3)}
                </div>
              </div>
              <div className="p-2 bg-gray-50 border-t border-gray-100">
                <p className="text-xs text-gray-600 truncate" title={result.chart_type}>
                  {result.chart_type}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default BaselineResultsGrid;
