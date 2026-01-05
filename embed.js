// embed.js - Gemini embeddings
import fs from "fs";
import "dotenv/config";
import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("❌ Missing GEMINI_API_KEY");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);

// New general embedding model for Gemini API
// Official docs list text-embedding-004 for text embeddings. :contentReference[oaicite:0]{index=0}
const embedModel = genAI.getGenerativeModel({
  model: "text-embedding-004",
});

async function main() {
  const qa = JSON.parse(fs.readFileSync("qa.json", "utf8"));
  const result = [];

  console.log(`Generating Gemini embeddings for ${qa.length} Q&A items...`);

  for (let i = 0; i < qa.length; i++) {
    const item = qa[i];
    const text = `${item.question}\n${item.answer}`;

    const embRes = await embedModel.embedContent(text);
    const embedding = embRes.embedding?.values || [];

    result.push({
      ...item,
      embedding,
    });

    if ((i + 1) % 50 === 0) {
      console.log(`Processed ${i + 1}/${qa.length}`);
    }
  }

  fs.writeFileSync(
    "qa_with_embeddings.json",
    JSON.stringify(result),
    "utf8"
  );
  console.log("✅ Done. Saved to qa_with_embeddings.json");
}

main().catch((err) => {
  console.error("Embedding error:", err);
  process.exit(1);
});
