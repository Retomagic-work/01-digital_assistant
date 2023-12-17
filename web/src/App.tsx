import { Provider } from "react-redux";
import AppNavigation from "./navigation/AppNavigation";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import store from "./store";
import { setAuthInitialization } from "./store/reducers/authReducer";

import "@fontsource-variable/montserrat";
import "./global.scss";

const isAuthInitialized = localStorage.getItem("isAuthInitialized") === "true";
if (isAuthInitialized) {
  store.dispatch(setAuthInitialization(true));
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <AppNavigation />
        {/* <ReactQueryDevtools /> */}
      </QueryClientProvider>
    </Provider>
  );
}

export default App;
