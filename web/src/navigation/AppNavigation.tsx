import { BrowserRouter, Route, Routes } from "react-router-dom";
import { DocumentsPage, HomePage, LoginPage } from "../pages";
import Layout from "../components/Layout";
import { useAppSelector } from "../hooks/store";

function AppNavigation() {
    const { isAuthInitialized } = useAppSelector((state) => state.auth);

    return (
        <BrowserRouter>
            <Routes>
                {!isAuthInitialized ? (
                    <Route path="/" element={<LoginPage />} />
                ) : (
                    <Route path="/" element={<Layout />}>
                        <Route index element={<HomePage />} />
                        <Route path="documents" element={<DocumentsPage />} />
                    </Route>
                )}
            </Routes>
        </BrowserRouter>
    );
}

export default AppNavigation;
