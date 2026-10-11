import { Navigate, createBrowserRouter } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AgentPage } from "./pages/Agent";
import { ChatPage } from "./pages/Chat";
import { ComparePage } from "./pages/Compare";
import { DashboardPage } from "./pages/Dashboard";
import { TuningPage } from "./pages/Tuning";

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <Navigate to="/chat" replace /> },
      { path: "/chat/:id?", element: <ChatPage /> },
      { path: "/agent/:runId?", element: <AgentPage /> },
      { path: "/compare", element: <ComparePage /> },
      { path: "/tuning", element: <TuningPage /> },
      { path: "/dashboard", element: <DashboardPage /> },
    ],
  },
]);
