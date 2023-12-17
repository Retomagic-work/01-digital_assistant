import {
  ChangeEvent,
  Dispatch,
  MouseEvent,
  SetStateAction,
  useCallback,
  useEffect,
  useState,
} from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import CrossIcon from "../../icons/CrossIcon";
import Button from "../../Button";
import { putDescriptionDocument } from "../../../api/documents";

import c from "./EditModal.module.scss";

const EditModal = ({
  value,
  id,
  setIsModal,
}: {
  value: string;
  id: number;
  setIsModal: Dispatch<SetStateAction<boolean>>;
}) => {
  const [text, setText] = useState<string>("");
  const queryClient = useQueryClient();

  const { mutateAsync } = useMutation<void, Error>({
    mutationFn: () => {
      return putDescriptionDocument(id, text);
    },
  });

  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    setText(event.target.value);
  };

  const handleClose = useCallback(
    (e: MouseEvent<HTMLDivElement> | MouseEvent<SVGSVGElement>) => {
      const target = e.target as HTMLElement;

      if (
        target.className === c.edit ||
        target.tagName === "path" ||
        target.tagName === "svg"
      )
        return setIsModal(false);
    },
    []
  );

  const handleChangeDescription = useCallback(async () => {
    await mutateAsync();
    await queryClient.refetchQueries({ queryKey: ["documents"] });
    setIsModal(false);
  }, [mutateAsync, text, setText, queryClient, id]);

  useEffect(() => {
    setText(value);
  }, [value]);

  return (
    <div className={c.edit} onClick={handleClose}>
      <div className={c.container}>
        <div className={c.header}>
          <span>Редактирование</span>
          <CrossIcon onClick={handleClose} />
        </div>
        <label className={c.body}>
          <textarea value={text} className={c.input} onChange={handleChange} />
        </label>
        <Button onClick={handleChangeDescription}>Обновить</Button>
      </div>
    </div>
  );
};

export default EditModal;
