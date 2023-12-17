import {
  Dispatch,
  MouseEvent,
  SetStateAction,
  useEffect,
  useState,
} from "react";

import StarIcon from "../icons/StarIcon";
import StarIconFill from "../icons/StarIconFill";
import { putRating } from "../../api/requests";

import c from "./RatingContainer.module.scss";
import { useMutation } from "@tanstack/react-query";
import { useDispatch } from "react-redux";
import { changeRating } from "../../store/reducers/requestsReducer";

const RatingContainer = ({
  id,
  rating,
  isDownload,
  setRating,
}: {
  id: number;
  rating: number | null;
  isDownload?: boolean;
  setRating?: Dispatch<SetStateAction<number | null>>;
}) => {
  const dispatch = useDispatch();
  const [count, setCount] = useState<number>(rating || 0);
  const [isHover, setIsHover] = useState<boolean>(true);

  const { mutateAsync: mutationRating } = useMutation<void, Error>({
    mutationFn: () => {
      const number = count || 0;
      return putRating(id, number);
    },
  });

  const handleClick = async () => {
    setCount(count);
    setIsHover(false);
    if (isDownload && setRating) {
      setRating(count);
      return;
    }
    dispatch(changeRating({ id, count }));
    await mutationRating();
  };

  const handleMouseEnter = (e: MouseEvent<HTMLAnchorElement>) => {
    const target = e.target as HTMLElement;

    if (target?.dataset.index) {
      setCount(Number(target?.dataset.index));
    }
  };
  const handleMouseOver = () => {
    if (isHover) {
      setCount(rating || 0);
    }
  };

  useEffect(() => {
    setCount(rating || 0);
    setIsHover(true);
  }, [rating, id]);

  return (
    <div className={c.container}>
      {[...Array(5).keys()].map((index) => {
        if (index + 1 <= count) {
          return (
            <div key={index} className={c.wrapper}>
              <StarIconFill />
              <a
                onClick={handleClick}
                className={c.link}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseOver}
                data-index={index + 1}
              ></a>
            </div>
          );
        }
        return (
          <div key={index} className={c.wrapper}>
            <StarIcon />
            <a
              onClick={handleClick}
              className={c.link}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseOver}
              data-index={index + 1}
            ></a>
          </div>
        );
      })}
    </div>
  );
};

export default RatingContainer;
