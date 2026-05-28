import { GoogleGenAI } from '@google/genai';

// Initialize the SDK using your Vite environment variable
const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_GEMINI_API_KEY });

/**
 * System guardrails to enforce role, boundaries, tone, and contextual parameters.
 */
const SYSTEM_INSTRUCTION = `
You are the built-in AI Wellness Coach for "MindMeter", an intelligent student burnout prevention platform.

YOUR MISSION:
Provide warm, emotionally safe, lightweight, non-overwhelming, and human-centered support to students navigating stress, academic load, or community pressures.

STRICT PARAMETERS & BOUNDARIES:
1. Role Lock: Act ONLY as a supportive companion, peer helper, or wellness coach. 
2. Non-Clinical Deflection: You are NOT a medical doctor, therapist, or psychiatrist. Never provide formal psychological diagnoses, medication advice, or clinical treatment plans. If a user asks for clinical/medical advice, gently deflect, validate their feelings, and remind them to speak with a healthcare professional or school counselor.
3. Relevant Scoping: Only answer topics relevant to student life, wellness, burnout, stress management, sleep, academic habits, social integrations, and lifestyle balance. If the user tries to ask completely irrelevant questions (e.g., programming bugs, fixing cars, general trivia), politely bring the focus back to their well-being.
4. Tone Style: Clean, premium, conversational, empathetic, and encouraging. Avoid clinical jargon, harsh formatting, and do not use emojis. Use natural spacing to lower cognitive load.
`;

/**
 * Sends a structured conversation log to Gemini and gets a context-locked response.
 * @param {Array} messageHistory - List of existing { sender, text } messages.
 * @returns {Promise<string>} Cleaned markdown string text response.
 */
export const askAICoach = async (messageHistory) => {
  try {
    // Format our simple state array into the structural schema required by Gemini
    const contents = messageHistory.map(msg => ({
      role: msg.sender === 'user' ? 'user' : 'model',
      parts: [{ text: msg.text }]
    }));

    // Call the high-performance, low-latency Flash model
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: contents,
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        // Temperature calibrated low for predictable, safe, and focused wellness responses
        temperature: 0.4,
        maxOutputTokens: 350,
      }
    });

    return response.text;
  } catch (error) {
    console.error("Gemini Platform Service Exception:", error);
    return "I am having trouble processing that right now. Let us take a breath and try again in a moment.";
  }
};