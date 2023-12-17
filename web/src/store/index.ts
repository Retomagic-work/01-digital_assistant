import { configureStore } from "@reduxjs/toolkit";
import authReducer from "./reducers/authReducer";
import requestsReducer from "./reducers/requestsReducer";

const store = configureStore({
  reducer: {
    auth: authReducer,
    requests: requestsReducer,
  },
});

export type AppDispatch = typeof store.dispatch;

export type RootState = ReturnType<typeof store.getState>;

export default store;
