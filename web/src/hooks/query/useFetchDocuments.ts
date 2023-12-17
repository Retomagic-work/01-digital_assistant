import { useQuery } from "@tanstack/react-query";

import { getDocuments } from "../../api/documents";
import { DocumentsData } from "../../types/Documents";

const useFetchDocuments = () => {
  return useQuery<DocumentsData[], null>({
    queryKey: ["documents"],
    queryFn: async () => {
      return getDocuments();
    },
  });
};

export default useFetchDocuments;
