import { NavLink } from "react-router-dom";
import Button from "../Button/Button";

import c from "./Header.module.scss";

const Header = () => {
  const logout = () => {
    localStorage.removeItem("isAuthInitialized");
    localStorage.removeItem("token");
    window.location.reload();
  };

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
          <div className={c.logout} onClick={logout}>
            Выйти из аккаунта
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
