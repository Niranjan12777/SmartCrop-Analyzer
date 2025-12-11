// server/index.js
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import multer from "multer";
import fetch from "node-fetch";
import path from "path";
import fs from "fs";
import jwt from "jsonwebtoken";
import bcrypt from "bcrypt";
import { fileURLToPath } from "url";
import FormData from "form-data";
import cors from "cors";
import pkg from "pg";
const { Pool } = pkg;

// ---------------------------------------------------
// PATHS
// ---------------------------------------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------
// EXPRESS APP
// ---------------------------------------------------
const app = express();
app.use(express.json());

app.use(
  cors({
    origin: "http://localhost:5173",
    credentials: true,
  })
);

// ---------------------------------------------------
// CONFIG
// ---------------------------------------------------
const SECRET = process.env.JWT_SECRET || "supersecret_change_me";
const FLASK_URL = process.env.FLASK_URL || "http://localhost:5001/predict";

const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

// ---------------------------------------------------
// DB
// ---------------------------------------------------
const pool = new Pool({
  user: process.env.PGUSER || "postgres",
  host: process.env.PGHOST || "localhost",
  database: process.env.PGDATABASE || "smart_agri",
  password: process.env.PGPASSWORD || "postgres",
  port: process.env.PGPORT || 5432,
});

// ---------------------------------------------------
// INITIALIZE DB
// ---------------------------------------------------
async function initDb() {
  await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
      );
  `);

  await pool.query(`
      CREATE TABLE IF NOT EXISTS results (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMP DEFAULT NOW(),
        health_image BYTEA,
        stress_image BYTEA,
        moisture_image BYTEA
      );
  `);
}
initDb().catch(console.error);

// ---------------------------------------------------
// AUTH HELPERS
// ---------------------------------------------------
async function hashPassword(pw) {
  return await bcrypt.hash(pw, 10);
}

async function verifyPassword(pw, hash) {
  return await bcrypt.compare(pw, hash);
}

function signJwt(payload) {
  return jwt.sign(payload, SECRET, { expiresIn: "7d" });
}

function authMiddleware(req, res, next) {
  const auth = req.headers.authorization;
  if (!auth) return res.status(401).json({ error: "missing token" });

  const token = auth.split(" ")[1];

  try {
    req.user = jwt.verify(token, SECRET);
    next();
  } catch {
    return res.status(401).json({ error: "invalid token" });
  }
}

// ---------------------------------------------------
// MULTER
// ---------------------------------------------------
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, file, cb) =>
    cb(null, Date.now() + "_" + file.originalname),
});
const upload = multer({ storage });

// ---------------------------------------------------
// SIGNUP
// ---------------------------------------------------
app.post("/signup", async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password)
    return res.status(400).json({ error: "missing fields" });

  const hash = await hashPassword(password);

  try {
    const result = await pool.query(
      `INSERT INTO users (username, password_hash)
       VALUES ($1,$2)
       RETURNING id, username`,
      [username, hash]
    );

    res.json({ user: result.rows[0] });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

// ---------------------------------------------------
// LOGIN
// ---------------------------------------------------
app.post("/login", async (req, res) => {
  const { username, password } = req.body;

  const r = await pool.query(
    `SELECT id, username, password_hash FROM users WHERE username=$1`,
    [username]
  );

  if (!r.rows.length) return res.status(401).json({ error: "invalid" });

  const user = r.rows[0];
  const ok = await verifyPassword(password, user.password_hash);

  if (!ok) return res.status(401).json({ error: "invalid" });

  res.json({ token: signJwt({ id: user.id, username: user.username }) });
});

// ---------------------------------------------------
// UPLOAD → FLASK → STORE BYTEA IMAGES
// ---------------------------------------------------
app.post("/upload", authMiddleware, upload.array("files"), async (req, res) => {
  const uploaded = req.files;
  const userId = req.user.id;

  if (!uploaded || uploaded.length === 0) {
    return res.status(400).json({ error: "No files uploaded" });
  }

  // Build form for Flask
  const formData = new FormData();

  const zipFile = uploaded.find((f) =>
    f.originalname.toLowerCase().endsWith(".zip")
  );

  if (zipFile) {
    formData.append("file", fs.createReadStream(zipFile.path));
  } else {
    uploaded.forEach((f) => {
      formData.append("files", fs.createReadStream(f.path));
    });
  }

  formData.append("user_id", userId);

  try {
    const response = await fetch(FLASK_URL, { method: "POST", body: formData });
    const result = await response.json();

    if (!response.ok) {
      return res.status(500).json({ error: "Prediction failed", details: result });
    }

    // Read images from Flask output paths → buffer
    const healthBytes = fs.readFileSync(result.health);
    const stressBytes = fs.readFileSync(result.stress);
    const moistureBytes = fs.readFileSync(result.moisture);

    const insert = await pool.query(
      `INSERT INTO results (user_id, health_image, stress_image, moisture_image)
       VALUES ($1,$2,$3,$4)
       RETURNING id, created_at`,
      [userId, healthBytes, stressBytes, moistureBytes]
    );

    // delete uploaded original files
    uploaded.forEach((f) => {
      try {
        fs.unlinkSync(f.path);
      } catch {}
    });

    res.json({
      ok: true,
      id: insert.rows[0].id,
      created_at: insert.rows[0].created_at,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------
// GET RESULTS (RETURN BASE64 IMAGES)
// ---------------------------------------------------
app.get("/results", authMiddleware, async (req, res) => {
  const r = await pool.query(
    `
    SELECT id, created_at,
           encode(health_image, 'base64') AS health_b64,
           encode(stress_image, 'base64') AS stress_b64,
           encode(moisture_image, 'base64') AS moisture_b64
    FROM results
    WHERE user_id=$1
    ORDER BY created_at DESC
  `,
    [req.user.id]
  );

  const results = r.rows.map((row) => ({
    id: row.id,
    created_at: row.created_at,
    health_url: `data:image/png;base64,${row.health_b64}`,
    stress_url: `data:image/png;base64,${row.stress_b64}`,
    moisture_url: `data:image/png;base64,${row.moisture_b64}`,
  }));

  res.json({ results });
});

// ---------------------------------------------------
// DELETE RESULT (BYTEA ONLY — no file deletion)
// ---------------------------------------------------
app.delete("/results/:id", authMiddleware, async (req, res) => {
  const id = req.params.id;

  const r = await pool.query(
    `SELECT id FROM results WHERE id=$1 AND user_id=$2`,
    [id, req.user.id]
  );

  if (!r.rows.length)
    return res.status(404).json({ error: "not found" });

  await pool.query(`DELETE FROM results WHERE id=$1`, [id]);

  res.json({ ok: true });
});

// ---------------------------------------------------
// START SERVER
// ---------------------------------------------------
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
