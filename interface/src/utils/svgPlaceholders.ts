import { HistoryContentPart, HistoryMessageContent } from '../types';

export const restoreSvgPlaceholders = (text: string, placeholderMap: Record<string, string>): string => {
  if (!text) return text;
  if (!placeholderMap || Object.keys(placeholderMap).length === 0) return text;

  let restored = text;
  for (const [placeholder, imageData] of Object.entries(placeholderMap)) {
    restored = restored.split(placeholder).join(imageData);
  }
  return restored;
};

export const restoreHistoryMessageContent = (
  content: HistoryMessageContent,
  placeholderMap: Record<string, string>
): HistoryMessageContent => {
  if (typeof content === 'string') {
    return restoreSvgPlaceholders(content, placeholderMap);
  }

  return content.map((item: HistoryContentPart) => {
    if (item.type === 'text' && typeof item.text === 'string') {
      return {
        ...item,
        text: restoreSvgPlaceholders(item.text, placeholderMap),
      };
    }
    return item;
  });
};
