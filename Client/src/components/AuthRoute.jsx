import React from "react";
import { Navigate } from "react-router-dom";

export default function AuthRoute({ children }) {
  const token = localStorage.getItem("token");

  if (token) {
    return <Navigate to="/prediction" replace />;
  }

  return children;
}
