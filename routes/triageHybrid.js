const express = require("express");
const router = express.Router();
const {
  classifyWithIntelligentRAG,
  isHealthcareRelated,
} = require("../langchain/hybridClassifier");
router.post("/", async (req, res) => {
  let { description } = req.body;
  // Basic validation
  if (!description || typeof description !== "string" || !description.trim()) {
    return res.status(400).json({ error: "Invalid symptom description." });
  }
  // :drop_of_blood: Simple NLP augmentation for 'bleeding' with no location
  if (
    description.toLowerCase().includes("bleeding") &&
    !description.match(
      /(arm|leg|head|chest|wound|cut|nose|mouth|face|finger|toe|abdomen|back|shoulder|foot|thigh|eye|ear)/i
    )
  ) {
    description +=
      " (Note: User mentioned bleeding but did not specify where. Might need follow-up.)";
  }
  try {
    // :bulb: Check if input is clearly medical-related
    const isHealth = await isHealthcareRelated(description);
    if (!isHealth) {
      return res.status(400).json({
        success: false,
        error:
          "Please provide a more specific symptom. Avoid vague or unrelated phrases.",
      });
    }
    // :brain: Hybrid RAG classifier with LangSmith trace
    const result = await classifyWithIntelligentRAG(description);
    // :white_check_mark: Success Response
    res.json({
      success: true,
      ...result,
    });
  } catch (err) {
    console.error(":rotating_light: Hybrid RAG classification error:", err);
    res.status(500).json({
      success: false,
      error: "Hybrid classification failed due to internal error.",
    });
  }
});
module.exports = router;





