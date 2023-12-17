import { DocumentsData } from "../../types/Documents";
import DocumentsTableRow from "../DocumentsTableRow/DocumentsTableRow";

import c from "./DocumentsTable.module.scss";
interface tableProps {
  headerData: string[];
  bodyData: DocumentsData[];
}

function DocumentsTable({ headerData, bodyData }: tableProps) {
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
          <DocumentsTableRow key={index} elem={elem} />
        ))}
      </div>
    </div>
  );
}
export default DocumentsTable;
