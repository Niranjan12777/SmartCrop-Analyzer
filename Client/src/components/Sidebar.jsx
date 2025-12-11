import React from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Sidebar(){
  const nav = useNavigate();
  function logout(){
    localStorage.removeItem("token");
    nav("/login");
  }
  return (
    <div style={{width:220, background:"#222", color:"#fff", minHeight:"100vh", padding:20}}>
      <h3>SmartCrop Analyzer</h3>
      <nav style={{display:"flex", flexDirection:"column", gap:10}}>
        <Link to="/" style={{color:"#fff"}}>Home</Link>
        <Link to="/prediction" style={{color:"#fff"}}>Prediction</Link>
        <Link to="/results" style={{color:"#fff"}}>Results</Link>
        <Link to="/about" style={{color:"#fff"}}>About</Link>
        <Link to="/contact" style={{color:"#fff"}}>Contact</Link>
        <button onClick={logout} style={{marginTop:20}}>Logout</button>
      </nav>
    </div>
  );
}
