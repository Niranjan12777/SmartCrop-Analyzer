import React, {useState} from "react";
import { useNavigate } from "react-router-dom";

export default function Login(){
  const [username,setUsername]=useState("");
  const [password,setPassword]=useState("");
  const nav = useNavigate();

  async function submit(e){
    e.preventDefault();
    const res = await fetch("http://localhost:4000/login", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({username, password})
    });
    const j = await res.json();
    if (j.token) {
      localStorage.setItem("token", j.token);
      nav("/prediction");
    } else {
      alert("Login failed");
    }
  }

  return (
    <div style={{maxWidth:400, margin:"40px auto"}}>
      <h2>Login</h2>
      <form onSubmit={submit}>
        <input placeholder="username" value={username} onChange={e=>setUsername(e.target.value)} />
        <br/>
        <input placeholder="password" type="password" value={password} onChange={e=>setPassword(e.target.value)} />
        <br/>
        <button type="submit">Login</button>
      </form>
      <p>New? <a href="/signup">Signup</a></p>
    </div>
  );
}
