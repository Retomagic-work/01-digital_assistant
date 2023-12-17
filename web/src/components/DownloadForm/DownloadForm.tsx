import { useState } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import classNames from "classnames";

import Button from "../Button";
import CopyIcon from "../icons/CopyIcon";
import CrossIcon from "../icons/CrossIcon";
import { createDocuments } from "../../api/documents";
import Loader from "../Loader";

import c from "./DownloadForm.module.scss";

const DownloadForm = () => {
  const queryClient = useQueryClient();
  const [uploadedFiles, setUploadedFiles] = useState<any>([]);

  const { getRootProps, getInputProps, isDragAccept } = useDropzone({
    onDrop: (acceptedFiles) => {
      setUploadedFiles(acceptedFiles);
    },
    maxFiles: 100,
    accept: {
      "application/pdf": [".pdf"],
    },
  });

  const classN = classNames(
    { [c.dragFilesContainer]: true },
    { [c.drag]: isDragAccept }
  );

  const { mutateAsync: mutationCreateDocuments, isPending } = useMutation<
    void,
    Error
  >({
    mutationFn: () => {
      return createDocuments(uploadedFiles);
    },
  });

  const handleDownload = async () => {
    try {
      await mutationCreateDocuments();
      await queryClient.refetchQueries({
        queryKey: ["documents"],
      });
      setUploadedFiles([]);
    } catch (error) {
      console.error(error);
    }
  };

  const handleDelete = (value: string) => {
    const filter = uploadedFiles.filter((file: any) => file.name !== value);
    setUploadedFiles(filter);
  };

  return (
    <div className={c.container}>
      <div className={classN} {...getRootProps()}>
        <span className={c.title}>Перетаскивайте файлы сюда</span>
        <span className={c.subtitle}>Ограничение 200 мб на файл</span>
        <Button>
          Выбрать файл
          <input
            type="file"
            accept=".pdf"
            multiple
            className={c.input}
            {...getInputProps()}
          />
        </Button>
      </div>
      <div className={c.filesContainer}>
        <span className={c.text}>
          Загрузите PDF-файлы и нажмите на кнопку "Загрузить"
        </span>
        <div className={c.uploads}>
          {uploadedFiles.map((file: any, index: number) => (
            <div key={index} className={c.upload}>
              <CopyIcon />
              {file.name}
              <CrossIcon
                className={c.cross}
                onClick={() => handleDelete(file.name)}
              />
            </div>
          ))}
        </div>
        {!isPending ? (
          <Button onClick={handleDownload}>Загрузить</Button>
        ) : (
          <div className={c.loaderWrapper}>
            <div className={c.loader}>
              <Loader />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DownloadForm;
