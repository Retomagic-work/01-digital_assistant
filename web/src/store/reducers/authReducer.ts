import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface AuthState {
    isAuthInitialized: boolean;
}

const initialState: AuthState = {
    isAuthInitialized: false,
}

export const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        setAuthInitialization: (state, action: PayloadAction<boolean>) => {
            state.isAuthInitialized = action.payload;
            localStorage.setItem('isAuthInitialized', action.payload.toString());
        },
    },
})

export const { setAuthInitialization } = authSlice.actions

export default authSlice.reducer
