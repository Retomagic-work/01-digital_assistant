import {
  Dispatch,
  MouseEvent,
  SetStateAction,
  useCallback,
  useState,
} from "react";
import CrossIcon from "../../icons/CrossIcon";
import RatingContainer from "../../RatingContainer";

import Button from "../../Button";

import c from "./DownloadModal.module.scss";

const DownloadModal = ({
  setIsModal,
}: {
  setIsModal: Dispatch<SetStateAction<boolean>>;
}) => {
  const [activeCategory, setActiveCategory] = useState<string>("Без рейтинга");

  const handleChange = (value: string) => {
    setActiveCategory(value);
  };

  const handleClose = useCallback(
    (e: MouseEvent<HTMLDivElement> | MouseEvent<SVGSVGElement>) => {
      const target = e.target as HTMLElement;

      if (
        target.className === c.edit ||
        target.className === c.cross ||
        target.parentElement?.className === c.cross ||
        target.parentElement?.parentElement?.className === c.cross
      )
        return setIsModal(false);
    },
    []
  );

  return (
    <div className={c.edit} onClick={handleClose}>
      <div className={c.container}>
        <div className={c.header}>
          <span>Скачать</span>
          <div className={c.cross}>
            <CrossIcon onClick={handleClose} className={c.cross} />
          </div>
        </div>
        <label className={c.label}>
          Дата с:
          <input type="date" className={c.input} placeholder="" />
        </label>
        <label className={c.label}>
          Дата по:
          <input type="date" className={c.input} placeholder="" />
        </label>
        <div className={c.ratingContainer}>
          <span>Рейтинг</span>
          <div className={c.buttons}>
            <Button
              onClick={() => handleChange("Без рейтинга")}
              className={`${c.button} ${
                activeCategory === "Без рейтинга" ? c.active : ""
              }`}
            >
              Без рейтинга
            </Button>
            <Button
              onClick={() => handleChange("С рейтингом")}
              className={`${c.button} ${
                activeCategory === "С рейтингом" ? c.active : ""
              }`}
            >
              С рейтингом
            </Button>
            {activeCategory === "С рейтингом" && (
              <RatingContainer id={0} rating={5} />
            )}
          </div>
        </div>
        <Button>Обновить</Button>
      </div>
    </div>
  );
};

export default DownloadModal;
