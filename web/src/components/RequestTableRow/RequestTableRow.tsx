import Clipboard from "../Clipboard";
import TextShort from "../TextShort";
import RatingContainer from "../RatingContainer";
import { formatDate } from "../../utils/formatDate";
import { RequestData } from "../../types/Requests";

import c from "./RequestTableRow.module.scss";

const RequestTableRow = ({ elem }: { elem: RequestData }) => {
  if (elem.id === 13) {
    console.log(elem.rating);
  }

  return (
    <div className={c.row}>
      <div className={c.textContainer}>
        <TextShort value={elem.data.text} />
        <div className={c.svg}>
          <Clipboard text={elem.data.text} />
        </div>
      </div>
      <div className={c.textContainer}>
        <TextShort value={elem.response.text} />
        <Clipboard text={elem.data.text} />
      </div>
      <span className={c.textContainer}>{elem.response.confidence * 100}%</span>
      <div className={c.rating}>
        <RatingContainer id={elem.id} rating={elem.rating} />
      </div>
      <span className={c.textContainer}>{formatDate(elem.created_at)}</span>
    </div>
  );
};

export default RequestTableRow;
