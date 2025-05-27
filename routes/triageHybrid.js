// routes/triageHybrid.js
const express = require("express");
const router = express.Router();
const {
  classifyWithHybridRAG,
  isHealthcareRelated,
} = require("../langchain/hybridClassifier");

router.post("/", async (req, res) => {
  //console.log("🛬 Incoming /triage-hybrid request"); // NEW LOG
  let { description } = req.body; // ✅ CHANGED from `const` to `let`
  if (!description || typeof description !== "string" || !description.trim()) {
    return res.status(400).json({ error: "Invalid symptom description." });
  }

  // 🔍 Prompt Engineering Middleware
  if (
    description.toLowerCase().includes("bleeding") &&
    !description.match(
      /(arm|leg|head|chest|wound|cut|nose|mouth|face|finger|toe|abdomen|back|shoulder|foot|thigh|eye|ear)/i
    )
  ) {
    description +=
      " (Note: User mentioned bleeding but did not specify where. Might need follow-up.)";
    //console.log("🩸 Augmented Description:", description); // ✅ Add this
  }

  try {
    const isHealth = await isHealthcareRelated(description);
    if (!isHealth) {
      return res.status(400).json({
        success: false,
        error: "Please provide a more specific symptom. Avoid vague inputs like 'not feeling well' or unrelated phrases.",
      });
    }

    const result = await classifyWithHybridRAG(description);
    res.json({
      success: true,
      ...result,
    });
  } catch (err) {
    console.error("Hybrid RAG classification error:", err);
    res.status(500).json({ error: "Hybrid classification failed." });
  }
});

module.exports = router;
