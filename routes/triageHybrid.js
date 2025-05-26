// routes/triageHybrid.js
const express = require("express");
const router = express.Router();
const {
  classifyWithHybridRAG,
  isHealthcareRelated,
} = require("../langchain/hybridClassifier");

router.post("/", async (req, res) => {
  const { description } = req.body;
  if (!description || typeof description !== "string" || !description.trim()) {
    return res.status(400).json({ error: "Invalid symptom description." });
  }

  try {
    const isHealth = await isHealthcareRelated(description);
    if (!isHealth) {
      return res.status(400).json({
        success: false,
        error: "Only healthcare-related prompts are allowed.",
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
