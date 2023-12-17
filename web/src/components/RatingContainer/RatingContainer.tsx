import { MouseEvent, useEffect, useState } from "react";

import StarIcon from "../icons/StarIcon";
import StarIconFill from "../icons/StarIconFill";
import { putRating } from "../../api/requests";

import c from "./RatingContainer.module.scss";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const RatingContainer = ({
  id,
  rating,
}: {
  id: number;
  rating: number | null;
}) => {
  const [count, setCount] = useState<number | null>(rating);

  const queryClient = useQueryClient();

  const { mutateAsync: mutationRating } = useMutation<void, Error>({
    mutationFn: () => {
      const number = count || 0;
      return putRating(id, number);
    },
  });

  const handleClick = async () => {
    await mutationRating();
    await queryClient.refetchQueries({
      queryKey: ["requests"],
    });
  };

  const handleMouseEnter = (e: MouseEvent<HTMLAnchorElement>) => {
    const target = e.target as HTMLElement;

    if (target?.dataset.index) {
      setCount(Number(target?.dataset.index));
    }
  };
  const handleMouseOver = () => {
    setCount(rating);
  };

  useEffect(() => {
    // if (rating) {
    //   setCount(rating);
    //   return;
    // }
  }, [rating]);

  return (
    <div className={c.container}>
      {[...Array(5).keys()].map((index) => {
        if (count && index + 1 <= count) {
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
