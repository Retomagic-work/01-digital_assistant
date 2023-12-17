import { useState } from "react";

import Arrow from "../icons/Arrow";

import c from "./TextShort.module.scss";

const TexShort = ({
  value,
  oneColumn,
}: {
  value: string;
  oneColumn?: boolean;
}) => {
  const [isDataShort, setIsDataShort] = useState<boolean>(true);

  const newValue = (text: string) => {
    if (!isDataShort) return text;
    if (!text) return "";
    if (oneColumn) {
      return text.length > 95 ? `${text.substring(0, 95)}...` : text;
    }
    return text.length > 130 ? `${text.substring(0, 130)}...` : text;
  };

  const handleClick = () => {
    setIsDataShort((prev) => !prev);
  };

  return (
    <div className={c.container} data-one-column={oneColumn}>
      <p className={c.text}>{newValue(value)}</p>
      {oneColumn
        ? value?.length > 95 && (
            <button className={c.button} onClick={handleClick}>
              {isDataShort ? (
                <>
                  Показать все
                  <Arrow />
                </>
              ) : (
                <>
                  Скрыть
                  <Arrow data-rotate={isDataShort} />
                </>
              )}
            </button>
          )
        : value?.length > 130 && (
            <button className={c.button} onClick={handleClick}>
              {isDataShort ? (
                <>
                  Показать все
                  <Arrow />
                </>
              ) : (
                <>
                  Скрыть
                  <Arrow data-rotate={isDataShort} />
                </>
              )}
            </button>
          )}
    </div>
  );
};

export default TexShort;
