import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";

import { authorizationUser } from "../../api/auth";
import Button from "../../components/Button";
import { setAuthInitialization } from "../../store/reducers/authReducer";

import c from "./LoginPage.module.scss";

function LoginPage() {
  const navigate = useNavigate();
  const [isError, setIsError] = useState<boolean>(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const dispatch = useDispatch();

  const handleClick = async () => {
    if (email.length < 3 || password.length < 3) return;

    try {
      const userData = await authorizationUser({ login: email, password }).then(
        (data) => {
          dispatch(setAuthInitialization(true));
          return data;
        }
      );

      localStorage.setItem("token", JSON.stringify(userData.access_token));
      navigate("/");
    } catch (error) {
      setIsError(true);
      console.error("Login failed:", error);
    }
  };

  return (
    <div className={c.login}>
      <div className={c.container}>
        <span className={c.title}>Log in</span>
        {isError && <div className={c.error}>Не верный логин или пароль</div>}
        <input
          className={c.input}
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter login here"
        />
        <input
          className={c.input}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter password here"
        />
        <Button className={c.button} onClick={handleClick}>
          Sign in
        </Button>
      </div>
    </div>
  );
}

export default LoginPage;
