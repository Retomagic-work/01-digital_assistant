import { Dispatch, SetStateAction } from "react";

import Button from "../Button";
import SearchForm from "../SearchForm";

import c from "./HeaderTable.module.scss";

const HeaderTable = ({
  title,
  setSearchText,
  isDownload,
  setIsModal,
}: {
  title: string;
  setSearchText: Dispatch<SetStateAction<string>>;
  isDownload?: boolean;
  setIsModal?: Dispatch<SetStateAction<boolean>>;
}) => {
  return (
    <div className={c.container}>
      <div className={c.left}>
        <p className={c.title}>{title}</p>
        {isDownload && setIsModal && (
          <Button onClick={() => setIsModal(true)}>Скачать</Button>
        )}
      </div>
      <SearchForm setSearchText={setSearchText} />
    </div>
  );
};

export default HeaderTable;
