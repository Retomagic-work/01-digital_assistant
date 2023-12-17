import { NavLink } from "react-router-dom";

import c from "./Header.module.scss";
import Button from "../Button/Button";

const Header = () => {
  return (
    <header className={c.header}>
      <div className="container">
        <div className={c.container}>
          <div className={c.buttons}>
            <NavLink
              to="/"
              className={({ isActive }) => (isActive ? c.active : "")}
            >
              <Button>Запросы</Button>
            </NavLink>
            <NavLink
              to="/documents"
              className={({ isActive }) => (isActive ? c.active : "")}
            >
              <Button>Документы</Button>
            </NavLink>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
