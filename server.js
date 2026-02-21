// server.js - AllMarkets Gemini backend

import express from "express";
import cors from "cors";
import fs from "fs";
import "dotenv/config";
import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("❌ Missing, GEMINI_API_KEY in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);

// Fast chat model for answers
const chatModel = genAI.getGenerativeModel({
  model: "gemini-2.0-flash",
});

// Embedding model for user questions
const embedModel = genAI.getGenerativeModel({
  model: "text-embedding-004",
});

// Load KB with embeddings
const knowledgeBase = JSON.parse(
  fs.readFileSync("qa_with_embeddings.json", "utf8")
);

// ---------- Helpers ----------
function cosineSim(a, b) {
  let dot = 0,
    aMag = 0,
    bMag = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    aMag += a[i] * a[i];
    bMag += b[i] * b[i];
  }
  if (aMag === 0 || bMag === 0) return 0;
  return dot / (Math.sqrt(aMag) * Math.sqrt(bMag));
}

function getTopN(vec, n = 3) {
  const scored = knowledgeBase.map((item) => ({
    item,
    score: cosineSim(vec, item.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, n);
}

// ---------- Express app ----------
const app = express();
app.use(cors());
app.use(express.json());

app.post("/allmarkets-chat", async (req, res) => {
  const question = (req.body.message || "").trim();

  if (!question) {
    return res.json({
      reply: "Please type your question about AllMarkets.",
    });
  }

  try {
    // 1) Embed user question
    const embRes = await embedModel.embedContent(question);
    const qVec = embRes.embedding?.values || [];

    // 2) Retrieve top relevant Q&A (top 3 for token efficiency)
    const top = getTopN(qVec, 3);
    const best = top[0];

    // (Optional) If similarity is extremely high, just return FAQ answer directly
    if (best && best.score >= 0.9) {
      return res.json({
        reply: best.item.answer,
      });
    }

    // 3) Build compact context for Gemini
    const context = top
      .map(
        (t, i) =>
          `Snippet ${i + 1} [${t.item.category}]:\nQ: ${t.item.question}\nA: ${t.item.answer}`
      )
      .join("\n\n");

    const prompt = [
      "You are the official Ai AllMarkets customer care assistant.",
      "Answer ONLY using the provided context below, and reason where nescessary but within context.",
      "If you are not sure, say you are not sure and direct the user to the Help/Contact page on allmarkets.org.",
      "",
      `User question: ${question}`,
      "",
      "Context:",
      context,
    ].join("\n");

    // 4) Ask Gemini with generationConfig to control token usage
    const result = await chatModel.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: 256, // cap response length
        temperature: 0.2, // more deterministic, less waffling
      },
    });

    const answer = result.response.text();

    return res.json({
      reply: answer,
      debug: {
        topScores: top.map((t) => t.score),
      },
    });
  } catch (err) {
    console.error("Gemini API error:", err);

    // Be explicit when quota is dead
    if (err.status === 429) {
      return res.status(429).json({
        reply:
          "Out of responses please try again later or contact AllMarkets support for assistance.",
      });
    }

    return res.status(500).json({
      reply:
        "I ran into a technical issue. Please try again later or contact AllMarkets support.",
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 AllMarkets Gemini bot backend running on port ${PORT}`);
});
