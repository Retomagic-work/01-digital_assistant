import { useState } from "react";
import ReactPaginate from "react-paginate";

import HeaderTable from "../../components/HeaderTable";
import RequestTable from "../../components/RequestTable";
import Arrow from "../../components/icons/Arrow";

import { DownloadModal } from "../../components/modals";
import useFetchRequests from "../../hooks/query/useFetchRequests";

import c from "./HomePage.module.scss";

const headerData = [
  "Запрос",
  "Ответ",
  "Уверенность",
  "Оценка ответа",
  "Создано",
];

interface PageClickEvent {
  selected: number;
}

function HomePage() {
  const [isModal, setIsModal] = useState(false);

  const itemsPerPage = 10;
  const [itemOffset, setItemOffset] = useState(0);
  const [searchText, setSearchText] = useState<string>("");

  const { data, isLoading } = useFetchRequests();

  if (!data) {
    return null;
  }

  const filterData = data.filter((elem) =>
    elem.data.text.toLowerCase().includes(searchText.toLowerCase())
  );

  const endOffset = itemOffset + itemsPerPage;
  const currentItems = filterData.slice(itemOffset, endOffset);
  const pageCount = Math.ceil(filterData.length / itemsPerPage);

  const handlePageClick = (event: PageClickEvent) => {
    const newOffset = (event.selected * itemsPerPage) % filterData.length;
    setItemOffset(newOffset);
    window.scrollTo(0, 0);
  };

  return (
    <section className={c.request}>
      <div className="container">
        <div className={c.container}>
          <HeaderTable
            title="Запросы"
            setSearchText={setSearchText}
            setIsModal={setIsModal}
            isDownload
          />
          {isModal && <DownloadModal setIsModal={setIsModal} />}
          {!isLoading && (
            <>
              <RequestTable bodyData={currentItems} headerData={headerData} />
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
          )}
        </div>
      </div>
    </section>
  );
}

export default HomePage;
