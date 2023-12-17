import c from "./Loader.module.scss";

function Loader() {
  return (
    <svg
      className={c.loader}
      width="34"
      height="34"
      viewBox="0 0 34 34"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g opacity="0.3">
        <path
          d="M17 3C24.7266 3 31 9.2734 31 17C31 24.7266 24.7266 31 17 31C9.2734 31 3 24.7266 3 17C3 9.2734 9.2734 3 17 3Z"
          stroke="#2E2C37"
          strokeWidth="5"
        />
      </g>
      <path
        d="M14.7108 30.8131C8.8145 29.8409 4.13392 25.1713 3.17902 19.2382"
        stroke="url(#paint0_linear_4754_9475)"
        strokeWidth="5"
        strokeLinecap="round"
      />
      <defs>
        <linearGradient
          id="paint0_linear_4754_9475"
          x1="-18.0987"
          y1="30.0602"
          x2="-17.2102"
          y2="17.0205"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0.111497" stopColor="#303381" />
          <stop offset="1" stopColor="#303381" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default Loader;
