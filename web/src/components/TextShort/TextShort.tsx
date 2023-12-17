import { useEffect, useState } from "react";

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
  const [dataShort, setDataShort] = useState<string>("");

  useEffect(() => {
    const newValue = (text: string) => {
      if (!isDataShort) return text;
      if (!text) return setDataShort("");
      if (oneColumn) {
        const value = text.length > 85 ? `${text.substring(0, 85)}...` : text;
        setDataShort(value);
        return;
      }
      const value = text.length > 85 ? `${text.substring(0, 85)}...` : text;
      setDataShort(value);
    };
    newValue(value);
  }, [value]);

  const handleClick = () => {
    setIsDataShort((prev) => !prev);
  };

  return (
    <div className={c.container} data-one-column={oneColumn}>
      <p className={c.text}>{dataShort}</p>
      {dataShort.length > 85 && (
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
