// gemini_sanity_test.js
import "dotenv/config";
import { GoogleGenerativeAI } from "@google/generative-ai";

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  console.error("❌ GEMINI_API_KEY missing in .env");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);

async function main() {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const result = await model.generateContent(
      "Respond with exactly this text: ALLMARKETS-GEMINI-OK"
    );
    console.log("Gemini response:", result.response.text());
  } catch (err) {
    console.error("Gemini test error:", err);
  }
}

main();
