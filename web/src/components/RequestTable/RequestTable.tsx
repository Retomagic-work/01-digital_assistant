import { RequestData } from "../../types/Requests";
import RequestTableRow from "../RequestTableRow/RequestTableRow";

import c from "./RequestTable.module.scss";

interface tableProps {
  headerData: string[];
  bodyData: RequestData[];
}

function RequestTable({ headerData, bodyData }: tableProps) {
  return (
    <div className={c.container}>
      <div className={c.table}>
        <div className={c.row}>
          {headerData.map((elem, index) => (
            <span key={index} className={c.titleContainer}>
              {elem}
            </span>
          ))}
        </div>
        {bodyData.map((elem, index) => (
          <RequestTableRow elem={elem} key={index} />
        ))}
      </div>
    </div>
  );
}
export default RequestTable;
