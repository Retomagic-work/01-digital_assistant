import { useQuery } from "@tanstack/react-query";

import { RequestData } from "../../types/Requests";
import { getRequestFilter } from "../../api/requests";

interface Params {
  dateTo: string;
  dateFrom: string;
  //   rating?: string;
}

const useFetchRequests = ({ dateTo, dateFrom }: Params) => {
  return useQuery<RequestData[], null>({
    queryKey: ["requests-filter"],
    queryFn: async () => {
      return getRequestFilter(dateFrom, dateTo);
    },
  });
};

export default useFetchRequests;
