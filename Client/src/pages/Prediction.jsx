import React, { useState } from "react";

export default function Prediction() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();

    if (!files || files.length === 0) {
      return alert("Please select a zip or folder of .tif files");
    }

    const token = localStorage.getItem("token");
    const fd = new FormData();

    let hasZip = false;

    // Check if any file is a ZIP
    for (let f of files) {
      if (f.name.toLowerCase().endsWith(".zip")) {
        fd.append("file", f);
        hasZip = true;
        break;
      }
    }

    if (!hasZip) {
      // Send multiple tifs as a zip-like batch
      for (let f of files) {
        fd.append("files", f);  // MULTIPLE FILE SUPPORT
      }
    }

    setLoading(true);

    try {
      const res = await fetch("http://localhost:4000/upload", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: fd
      });

      const j = await res.json();
      setLoading(false);

      if (res.ok) {
        alert("Prediction done and saved to Results");
      } else {
        alert("Failed: " + JSON.stringify(j));
      }
    } catch (err) {
      setLoading(false);
      alert("Error: " + err.message);
    }
  }

  return (
    <div style={{justifyContent: "center"}}>
      <h2>Prediction</h2>

      <form onSubmit={submit}>
        <input
          type="file"
          multiple
          directory=""
          webkitdirectory=""
          onChange={(e) => setFiles(e.target.files)}
        />

        <br />
        <button style={{width: "200px", height: "30px", marginTop: "5px" }} disabled={loading} type="submit">
          {loading ? "Processing..." : "Start Prediction"}
        </button>
      </form>

      <p>
        Upload options:<br />
        ✔ ZIP containing .tif bands<br />
        ✔ A whole folder with multiple band .tifs<br />
        ✔ Multiple individual .tif files<br />
      </p>
      <span>Navigate to <a href="https://browser.dataspace.copernicus.eu/" target="_blank">Copernicus</a> to select and download the specific region</span>
    </div>
  );
}
