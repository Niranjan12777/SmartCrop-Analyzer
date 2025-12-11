import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Prediction from "./pages/Prediction";
import Results from "./pages/Results";
import Sidebar from "./components/Sidebar";

import ProtectedRoute from "./components/ProtectedRoute";
import AuthRoute from "./components/AuthRoute";

import { Outlet } from "react-router-dom";

function Layout() {
  return (
    <div style={{ display: "flex" }}>
      <Sidebar />
      <div style={{ flex: 1, padding: 20 }}>
        <Outlet />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>

      {/* Public (unauthenticated) routes */}
      <Route
        path="/login"
        element={
          <AuthRoute>
            <Login />
          </AuthRoute>
        }
      />

      <Route
        path="/signup"
        element={
          <AuthRoute>
            <Signup />
          </AuthRoute>
        }
      />

      {/* Protected parent route with Sidebar */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        {/* Child routes (no nested <Routes> needed) */}
        <Route path="prediction" element={<Prediction />} />
        <Route path="results" element={<Results />} />
        <Route path="about" element={<div>About</div>} />
        <Route path="contact" element={<div>Contact</div>} />
        <Route index element={<Navigate to="prediction" replace />} />
      </Route>

      {/* Unknown route fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
