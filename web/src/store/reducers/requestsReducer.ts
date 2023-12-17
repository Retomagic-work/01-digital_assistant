import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { RequestData } from "../../types/Requests";
import { act } from "react-dom/test-utils";

interface AuthState {
  request: RequestData[];
}

const initialState: AuthState = {
  request: [],
};

export const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    setRequest: (state, action: PayloadAction<RequestData[]>) => {
      state.request = action.payload;
    },
    changeRating: (
      state,
      action: PayloadAction<{ id: number; count: number }>
    ) => {
      state.request = state.request.map((elem) => {
        if (elem.id === action.payload.id) {
          elem.rating === action.payload.count;
          return elem;
        }
        return elem;
      });
    },
  },
});

export const { setRequest, changeRating } = authSlice.actions;

export default authSlice.reducer;
