import { useState, Dispatch, SetStateAction } from "react";
import Button from "../Button/Button";
import SearchIcon from "../icons/SearchIcon";

import c from "./SearchForm.module.scss";

const SearchForm = ({
  setSearchText,
}: {
  setSearchText: Dispatch<SetStateAction<string>>;
}) => {
  const [inputValue, setInputValue] = useState("");

  const handleSearch = () => {
    setSearchText(inputValue);
  };

  return (
    <div className={c.container}>
      <input
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        type="text"
        placeholder="Поиск..."
        className={c.input}
      />
      <Button onClick={handleSearch}>
        <SearchIcon />
      </Button>
    </div>
  );
};

export default SearchForm;
