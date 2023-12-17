import { ButtonHTMLAttributes } from "react";

import c from "./Button.module.scss";
import classNames from "classnames";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  className?: string;
}

const Button = ({ children, className, ...props }: ButtonProps) => {
  const classN = classNames(
    { [c.button]: true },
    className ? { [className]: true } : {}
  );
  return (
    <button className={classN} type="button" {...props}>
      {children}
    </button>
  );
};

export default Button;
