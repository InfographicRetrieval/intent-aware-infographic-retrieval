import { ChatMessage, ChartTypeNode } from "../types";

export type ExtractedSvgItem = {
  key: string;
  messageId: string;
  timestamp: Date;
  svgCode: string;
};

export const extractAllSvgCodes = (messages: ChatMessage[]): ExtractedSvgItem[] => {
  const items: ExtractedSvgItem[] = [];
  const seenByMessageAndSvg = new Set<string>();

  const normalizeSvgForDedup = (svgCode: string): string => svgCode.replace(/\r\n/g, "\n").trim();

  const pushSvgItem = (messageId: string, timestamp: Date, svgCode: string, index: number): boolean => {
    const normalizedSvg = normalizeSvgForDedup(svgCode);
    if (!normalizedSvg) return false;

    const dedupKey = `${messageId}__${normalizedSvg}`;
    if (seenByMessageAndSvg.has(dedupKey)) return false;
    seenByMessageAndSvg.add(dedupKey);

    items.push({ key: `${messageId}__svg__${index}`, messageId, timestamp, svgCode: normalizedSvg });
    return true;
  };

  const fencedRe = /```svg\s*([\s\S]*?)```/gi;
  const rawSvgRe = /<svg\b[\s\S]*?<\/svg>/gi;

  for (const msg of messages) {
    if (msg.type !== "assistant" || !msg.content) continue;

    const text = msg.content;

    if (!text) continue;

    const messageId = msg.id;
    let index = 0;
    let hasFencedMatch = false;

    fencedRe.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = fencedRe.exec(text)) !== null) {
      hasFencedMatch = true;
      if (pushSvgItem(messageId, msg.timestamp, match[1] || "", index)) {
        index += 1;
      }
    }

    if (hasFencedMatch) continue;

    rawSvgRe.lastIndex = 0;
    while ((match = rawSvgRe.exec(text)) !== null) {
      if (pushSvgItem(messageId, msg.timestamp, match[0] || "", index)) {
        index += 1;
      }
    }
  }

  items.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  return items;
};

export const getAllChartTypesUnderNode = (node: ChartTypeNode): string[] => {
  const result: string[] = [...node.chart_types];
  node.children.forEach((child) => {
    result.push(...getAllChartTypesUnderNode(child));
  });
  return result;
};

export const getAllChildNodeNames = (node: ChartTypeNode): string[] => {
  const result = [node.name];
  node.children.forEach((child) => {
    result.push(...getAllChildNodeNames(child));
  });
  return result;
};

export const findChartTypeNodeByName = (nodes: ChartTypeNode[], name: string): ChartTypeNode | null => {
  for (const node of nodes) {
    if (node.name === name) return node;
    const found = findChartTypeNodeByName(node.children, name);
    if (found) return found;
  }
  return null;
};

export const getSelectedChartTypes = (nodes: ChartTypeNode[], selectedNodeNames: Set<string>): string[] => {
  if (selectedNodeNames.size === 0) return [];

  const allTypes = new Set<string>();
  selectedNodeNames.forEach((nodeName) => {
    const node = findChartTypeNodeByName(nodes, nodeName);
    if (!node) return;
    getAllChartTypesUnderNode(node).forEach((type) => allTypes.add(type));
  });

  return Array.from(allTypes);
};
