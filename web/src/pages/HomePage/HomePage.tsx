import { useEffect, useState } from "react";
import ReactPaginate from "react-paginate";

import { useDispatch } from "react-redux";
import HeaderTable from "../../components/HeaderTable";
import Loader from "../../components/Loader";
import RequestTable from "../../components/RequestTable";
import Arrow from "../../components/icons/Arrow";
import { DownloadModal } from "../../components/modals";
import useFetchRequests from "../../hooks/query/useFetchRequests";
import { useAppSelector } from "../../hooks/store";
import { setRequest } from "../../store/reducers/requestsReducer";

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
  const dispatch = useDispatch();
  const { request } = useAppSelector((state) => state.requests);

  const itemsPerPage = 60;
  const [itemOffset, setItemOffset] = useState(0);
  const [searchText, setSearchText] = useState<string>("");

  const { data, isLoading } = useFetchRequests();

  useEffect(() => {
    if (data) {
      dispatch(setRequest(data));
    }
  }, [data]);

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

  const filterData = request
    .filter((elem) =>
      elem.data.text.toLowerCase().includes(searchText.toLowerCase())
    )
    .reverse();

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
          {currentItems.length > 0 ? (
            <>
              <RequestTable bodyData={currentItems} headerData={headerData} />
              <ReactPaginate
                className="reactPaginate"
                breakLabel="..."
                nextLabel={<Arrow />}
                onPageChange={(event) => {
                  handlePageClick(event);
                  window.scrollTo(0, 0);
                }}
                pageRangeDisplayed={5}
                pageCount={pageCount}
                previousLabel={<Arrow />}
                renderOnZeroPageCount={null}
              />
            </>
          ) : (
            <div className={c.text}>Запросов не найдено</div>
          )}
        </div>
      </div>
    </section>
  );
}

export default HomePage;
