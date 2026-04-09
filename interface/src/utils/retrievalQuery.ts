import { RetrievalAspectKey, RetrievalAspectValue, RetrievalQuerySpec } from '../types';

export const RETRIEVAL_ASPECT_ORDER: RetrievalAspectKey[] = [
  'chart_type',
  'content',
  'layout',
  'style',
  'illustration',
];

const DEFAULT_ASPECT_VALUE: RetrievalAspectValue = {
  query: '',
  weight: 0,
};

const clampWeight = (weight: number): number => {
  if (!Number.isFinite(weight)) return 0;
  const normalized = Math.round(weight * 10) / 10;
  return Math.min(1, Math.max(0, normalized));
};

export const createEmptyRetrievalSpec = (): RetrievalQuerySpec => ({
  chart_type: { ...DEFAULT_ASPECT_VALUE },
  content: { ...DEFAULT_ASPECT_VALUE },
  layout: { ...DEFAULT_ASPECT_VALUE },
  style: { ...DEFAULT_ASPECT_VALUE },
  illustration: { ...DEFAULT_ASPECT_VALUE },
});

export const normalizeRetrievalSpec = (raw: unknown): RetrievalQuerySpec => {
  const base = createEmptyRetrievalSpec();
  if (!raw || typeof raw !== 'object') {
    return base;
  }

  for (const key of RETRIEVAL_ASPECT_ORDER) {
    const value = (raw as Record<string, unknown>)[key];
    if (!value || typeof value !== 'object') continue;

    base[key] = {
      query: typeof (value as Record<string, unknown>).query === 'string'
        ? (value as Record<string, string>).query
        : '',
      weight: clampWeight(Number((value as Record<string, unknown>).weight ?? 0)),
    };
  }

  return base;
};

export const parseRetrievalSpec = (query: string | null | undefined): RetrievalQuerySpec | null => {
  if (!query) return null;
  try {
    return normalizeRetrievalSpec(JSON.parse(query));
  } catch {
    return null;
  }
};

export const serializeRetrievalSpec = (spec: RetrievalQuerySpec): string =>
  JSON.stringify(normalizeRetrievalSpec(spec), null, 0);

export const updateRetrievalAspect = (
  spec: RetrievalQuerySpec,
  key: RetrievalAspectKey,
  patch: Partial<RetrievalAspectValue>
): RetrievalQuerySpec => ({
  ...spec,
  [key]: {
    ...spec[key],
    ...patch,
    query: typeof patch.query === 'string' ? patch.query : spec[key].query,
    weight: patch.weight === undefined ? spec[key].weight : clampWeight(patch.weight),
  },
});

export const adjustRetrievalAspectWeight = (
  spec: RetrievalQuerySpec,
  key: RetrievalAspectKey,
  delta: number
): RetrievalQuerySpec =>
  updateRetrievalAspect(spec, key, {
    weight: clampWeight(spec[key].weight + delta),
  });

export const retrievalSpecsEqual = (a: RetrievalQuerySpec | null, b: RetrievalQuerySpec | null): boolean => {
  if (!a || !b) return a === b;
  return serializeRetrievalSpec(a) === serializeRetrievalSpec(b);
};
