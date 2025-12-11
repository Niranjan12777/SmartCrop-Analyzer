// Results.jsx
import React, { useEffect, useState } from "react";

function Thumbnail({ src, onClick, onDelete }) {
  return (
    <div style={{ width: 200, margin: 8 }}>
      <img
        src={src}
        style={{ width: "100%", height: 120, objectFit: "cover", cursor: "pointer" }}
        onClick={onClick}
        alt=""
      />
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        <button onClick={onDelete}>Delete</button>
      </div>
    </div>
  );
}

function ResultItem({ r, onDelete }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h5>{new Date(r.created_at).toLocaleString()}</h5>

      <div style={{ display: "flex" }}>
        <Thumbnail src={r.health_url} onClick={() => window.open(r.health_url)} onDelete={onDelete} />
        <Thumbnail src={r.stress_url} onClick={() => window.open(r.stress_url)} onDelete={onDelete} />
        <Thumbnail src={r.moisture_url} onClick={() => window.open(r.moisture_url)} onDelete={onDelete} />
      </div>
    </div>
  );
}

export default function Results() {
  const [results, setResults] = useState([]);
  const token = localStorage.getItem("token");

  async function fetchResults() {
    const res = await fetch("http://localhost:4000/results", {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) {
      console.error("Failed to fetch results", res.status);
      return;
    }
    const j = await res.json();
    setResults(j.results || []);
  }

  useEffect(() => {
    fetchResults();
  }, []);

  async function handleDelete(id) {
    if (!confirm("Delete?")) return;
    const token = localStorage.getItem("token");
    await fetch(`http://localhost:4000/results/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` }
    });
    fetchResults();
  }

  return (
    <div>
      <h2>Your Results</h2>
      <div style={{ display: "flex", flexWrap: "wrap" }}>
        {results.map((r) => (
          <ResultItem key={r.id} r={r} onDelete={() => handleDelete(r.id)} />
        ))}
      </div>
    </div>
  );
}
