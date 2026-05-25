// 공용 타입 정의.

export type TranslateDirection = "j2s" | "s2j";

export interface EmotionScores {
  [label: string]: number;
}

export interface EmotionResult {
  label: string;
  label_id: number;
  scores: EmotionScores;
}

export interface TranslateResult {
  translation: string;
}

export interface Place {
  id: string;
  name: string;
  category: string;
  address: string;
  lat: number;
  lng: number;
  phone?: string;
  rating?: number;
  description?: string;
  /** kakao place url (길찾기/상세) */
  placeUrl?: string;
}

export type AttractionCategory = "자연" | "문화" | "체험" | "포토";

export interface Attraction {
  id: string;
  name: string;
  category: AttractionCategory;
  address: string;
  lat: number;
  lng: number;
  description?: string;
  duration?: string; // 권장 소요시간
}

export interface RoutePlanStop {
  name: string;
  lat?: number;
  lng?: number;
  time?: string;
  note?: string;
}

export interface RoutePlanDay {
  day: number;
  title?: string;
  stops: RoutePlanStop[];
}

export interface RoutePlan {
  summary: string;
  days: RoutePlanDay[];
  mock?: boolean;
}
