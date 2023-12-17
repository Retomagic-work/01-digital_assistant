export interface RequestData {
  id: number;
  data: {
    text: string;
  };
  response: {
    confidence: number;
    text: string;
  };
  created_at: string;
  rating: number | null;
}
