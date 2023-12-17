import { useMemo, useState } from "react";

import CheckIcon from "../icons/CheckIcon";
import CopyRightIcon from "../icons/CopyRightIcon";

import c from "./Clipboard.module.scss";

interface ClipboardProps {
  text: string;
}

const Clipboard = ({ text }: ClipboardProps) => {
  const [isActive, setIsActive] = useState<boolean>(false);

  const saveClipboard = (value: string) => {
    navigator.clipboard.writeText(value).then(() => {
      setIsActive(true);
      setTimeout(() => {
        setIsActive(false);
      }, 1000);
    });
  };

  const CopyIcon = useMemo(() => {
    return isActive ? (
      <CheckIcon width={24} height={24} />
    ) : (
      <CopyRightIcon onClick={() => saveClipboard(text)} />
    );
  }, [isActive]);

  return <div className={c.container}>{CopyIcon}</div>;
};

export default Clipboard;
