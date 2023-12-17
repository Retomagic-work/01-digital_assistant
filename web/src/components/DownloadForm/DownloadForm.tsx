import { useState } from "react";
import { useDropzone } from "react-dropzone";

import Button from "../Button";
import CopyIcon from "../icons/CopyIcon";
import CrossIcon from "../icons/CrossIcon";
import { createDocuments } from "../../api/documents";

import c from "./DownloadForm.module.scss";

const DownloadForm = () => {
  const [uploadedFiles, setUploadedFiles] = useState<any>([]);
  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      setUploadedFiles(acceptedFiles);
    },
    maxFiles: 100,
    accept: {
      "application/pdf": [".pdf"],
    },
  });

  const handleDownload = async () => {
    try {
      const downloadData = await createDocuments(uploadedFiles);
      return downloadData;
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
      <div className={c.dragFilesContainer} {...getRootProps()}>
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
            onChange={(e) => console.log(e.target.files)}
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
        <Button onClick={handleDownload}>Загрузить</Button>
      </div>
    </div>
  );
};

export default DownloadForm;
