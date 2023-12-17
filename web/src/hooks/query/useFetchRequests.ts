import { useQuery } from "@tanstack/react-query";

import { RequestData } from "../../types/Requests";
import { getRequest } from "../../api/requests";

const useFetchRequests = () => {
  return useQuery<RequestData[], null>({
    queryKey: ["requests"],
    queryFn: async () => {
      return getRequest();
    },
  });
};

export default useFetchRequests;
