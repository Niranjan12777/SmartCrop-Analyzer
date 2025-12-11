import React, {useState} from "react";
import { useNavigate } from "react-router-dom";

export default function Signup(){
  const [username,setUsername]=useState("");
  const [password,setPassword]=useState("");
  const nav = useNavigate();

  async function submit(e){
    e.preventDefault();
    const res = await fetch("http://localhost:4000/signup", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({username, password})
    });
    if (res.ok) {
      alert("Signup success. Please login.");
      nav("/login");
    } else {
      const j = await res.json();
      alert("Signup failed: " + (j.error || JSON.stringify(j)));
    }
  }

  return (
    <div style={{maxWidth:400, margin:"40px auto"}}>
      <h2>Signup</h2>
      <form onSubmit={submit}>
        <input placeholder="username" value={username} onChange={e=>setUsername(e.target.value)} />
        <br/>
        <input placeholder="password" type="password" value={password} onChange={e=>setPassword(e.target.value)} />
        <br/>
        <button type="submit">Signup</button>
      </form>
    </div>
  );
}
