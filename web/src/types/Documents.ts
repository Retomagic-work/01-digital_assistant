export interface DocumentsData {
  id: number;
  description: string;
  link: string;
  created_at: string;
}

export interface FileData {
  path: string;
  lastModified: number;
  name: string;
  size: number;
  type: string;
}
