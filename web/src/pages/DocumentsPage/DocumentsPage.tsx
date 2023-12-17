import { useState } from "react";
import ReactPaginate from "react-paginate";

import DownloadForm from "../../components/DownloadForm";
import HeaderTable from "../../components/HeaderTable";
import DocumentsTable from "../../components/DocumentsTable";
import Arrow from "../../components/icons/Arrow";
import useFetchDocuments from "../../hooks/query/useFetchDocuments";
import Loader from "../../components/Loader";

import c from "./DocumentsPage.module.scss";

const headerData = ["Описание", "Создано", "Ссылка"];

const DocumentsPage = () => {
  const itemsPerPage = 10;
  const [itemOffset, setItemOffset] = useState(0);
  const [searchText, setSearchText] = useState<string>("");

  const { data, isLoading } = useFetchDocuments();

  if (isLoading) {
    return (
      <div className="loaderWrapper">
        <div className="loader">
          <Loader />
        </div>
      </div>
    );
  }

  if (!data) return null;

  const filterData = data
    .filter((elem) =>
      elem.description.toLowerCase().includes(searchText.toLowerCase())
    )
    .reverse();

  const endOffset = itemOffset + itemsPerPage;
  const currentItems = filterData.slice(itemOffset, endOffset);
  const pageCount = Math.ceil(filterData.length / itemsPerPage);

  const handlePageClick = ({ selected }: { selected: number }) => {
    const newOffset = (selected * itemsPerPage) % filterData.length;
    setItemOffset(newOffset);
    window.scrollTo(0, 0);
  };
  console.log(currentItems);

  return (
    <div className={c.documents}>
      <div className="container">
        <div className={c.container}>
          <DownloadForm />
          <HeaderTable title="Документы" setSearchText={setSearchText} />
          {currentItems.length > 0 ? (
            <>
              <DocumentsTable bodyData={currentItems} headerData={headerData} />
              <ReactPaginate
                className="reactPaginate"
                breakLabel="..."
                nextLabel={<Arrow />}
                onPageChange={handlePageClick}
                pageRangeDisplayed={5}
                pageCount={pageCount}
                previousLabel={<Arrow />}
                renderOnZeroPageCount={null}
              />
            </>
          ) : (
            <div className={c.text}>Документов не найдено.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentsPage;
