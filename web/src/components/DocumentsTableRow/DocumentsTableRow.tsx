import { useState } from "react";

import Button from "../Button";
import TextShort from "../TextShort";
import EditIcon from "../icons/EditIcon";
import { EditModal } from "../modals";
import { DocumentsData } from "../../types/Documents";
import { formatDate } from "../../utils/formatDate";

import c from "./DocumentsTableRow.module.scss";

const DocumentsTableRow = ({ elem }: { elem: DocumentsData }) => {
  const [isModal, setIsModal] = useState<boolean>(false);
  const handleOpenModalEdit = () => {
    setIsModal(true);
  };

  return (
    <div className={c.row}>
      <div className={c.textContainer}>
        <EditIcon onClick={handleOpenModalEdit} />
        <TextShort value={elem.description} oneColumn />
      </div>
      <span className={c.dateContainer}>{formatDate(elem.created_at)}</span>
      <a href={elem.link} target="_blank">
        <Button>Открыть PDF-файл</Button>
      </a>
      {isModal && (
        <EditModal
          value={elem.description}
          id={elem.id}
          setIsModal={setIsModal}
        />
      )}
    </div>
  );
};

export default DocumentsTableRow;
