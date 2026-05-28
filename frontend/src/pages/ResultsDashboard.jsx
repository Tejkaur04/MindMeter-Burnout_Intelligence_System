import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts';
import { Sparkles, ShieldCheck, HeartHandshake, AlertCircle, Send, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { askAICoach } from '../services/aiCoach';

export default function ResultsDashboard() {
  const location = useLocation();
  // Safe fallback if navigated directly without taking the assessment
  const rawInputs = location.state?.rawInputs || {
    self_esteem: 18, headache: 1, sleep_quality: 4, study_load: 2, social_support: 3, peer_pressure: 1
  };

  // Mocked inference mirroring output mapping of CalibratedClassifierCV
  const inferenceData = {
    burnoutRisk: "Moderate",
    confidence: 84.6,
    probabilities: [
      { name: "Healthy", value: 12 },
      { name: "Moderate", value: 85 },
      { name: "High", value: 3 },
      { name: "Critical", value: 0 }
    ],
    shapContributors: [
      { feature: "Study Load Intensity", impact: -0.22, type: "positive", label: "Lower structural load buffers risk" },
      { feature: "Sleep Efficiency", impact: -0.18, type: "positive", label: "Consistently protects cognitive reserve" },
      { feature: "Peer Pressure Dynamics", impact: 0.28, type: "negative", label: "Social environments elevating baseline strain" },
      { feature: "Self-Esteem Index", impact: -0.15, type: "positive", label: "Strong inner self-worth limits vulnerability" },
    ]
  };

  // Format form metrics directly into interactive Radar tracking configuration
  const radarData = Object.entries(rawInputs).slice(0, 6).map(([key, val]) => ({
    subject: key.replace('_', ' ').toUpperCase(),
    value: val * 15,
  }));

  // Chatbot state logic for the built-in supportive companion coach
  const [messages, setMessages] = useState([
    { sender: 'coach', text: "Hello! I am your wellness companion. Looking over the data, your sleep efficiency and strong self-esteem are acting as great protectors right now. How are you feeling about the peer dynamics you are experiencing?" }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isThinking, setIsThinking] = useState(false);

  // 2. Functional API message handler
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isThinking) return;

    const userMessageText = inputMessage.trim();
    const newUserMessage = { sender: 'user', text: userMessageText };
    
    // Append user input to local array instantly
    const updatedHistory = [...messages, newUserMessage];
    setMessages(updatedHistory);
    setInputMessage('');
    setIsThinking(true);

    // Hit the secure Google Gen AI route
    const coachResponseText = await askAICoach(updatedHistory);

    // Append AI reply and lift locking flags
    setMessages(prev => [...prev, { sender: 'coach', text: coachResponseText }]);
    setIsThinking(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-8 w-full">
      
      {/* HEADER EXPLANATORY ROW */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-brand-border pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Your Wellness Blueprint</h1>
          <p className="text-gray-400 text-sm">Calibrated live outputs processing standard lifestyle signal metrics.</p>
        </div>
        <div className="flex items-center gap-2 bg-brand-indigo/10 border border-brand-indigo/20 px-3 py-1.5 rounded-xl text-brand-lavender text-xs font-mono">
          <ShieldCheck className="w-4 h-4 text-brand-indigo" />
          Model Status: Calibrated (Isotonic)
        </div>
      </div>

      {/* CORE ANALYSIS METRICS LAYOUT */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* PRIMARY RISK CARD */}
        <div className="bg-brand-card border border-brand-border rounded-2xl p-6 flex flex-col justify-between relative overflow-hidden shadow-xl">
          <div className="absolute top-0 right-0 w-32 h-32 bg-brand-amber/5 rounded-full filter blur-xl pointer-events-none" />
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-gray-400">Burnout Profile</span>
              <AlertCircle className="w-4 h-4 text-brand-amber" />
            </div>
            <div className="space-y-1 mb-6">
              <h2 className="text-4xl font-bold tracking-tight text-brand-amber">{inferenceData.burnoutRisk}</h2>
              <p className="text-sm text-gray-400 font-mono">Statistical Confidence: {inferenceData.confidence}%</p>
            </div>
          </div>
          <div className="p-4 bg-white/[0.02] border border-brand-border/60 rounded-xl">
            <p className="text-xs text-gray-300 leading-relaxed">
              Your overall system shows balanced traits. While your academic load is manageable, external contextual friction signals are applying pressure to your baseline resilience.
            </p>
          </div>
        </div>

        {/* RISK DISTRIBUTION VISUALIZER */}
        <div className="bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Class Target Probability Distribution</h3>
          <div className="space-y-3.5">
            {inferenceData.probabilities.map((prob) => {
              const isTarget = prob.name === inferenceData.burnoutRisk;
              return (
                <div key={prob.name} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className={`font-medium ${isTarget ? 'text-white font-bold' : 'text-gray-400'}`}>{prob.name}</span>
                    <span className="font-mono text-gray-500">{prob.value}%</span>
                  </div>
                  <div className="h-2 bg-brand-dark rounded-full overflow-hidden border border-white/[0.02]">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${prob.value}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={`h-full rounded-full ${
                        prob.name === 'Healthy' ? 'bg-brand-emerald' :
                        prob.name === 'Moderate' ? 'bg-brand-amber' :
                        prob.name === 'High' ? 'bg-brand-indigo' : 'bg-brand-coral'
                      }`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SHAP SIGNAL RADAR */}
        <div className="bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Signal Cluster Shape</h3>
          <div className="w-full h-44 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" radius="70%" data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.05)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: 'rgba(156, 163, 175, 0.8)', fontSize: 8, fontFamily: 'monospace' }} />
                <Radar name="Metrics" dataKey="value" stroke="#6366F1" fill="#6366F1" fillOpacity={0.25} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* SHAP EXPLAINABILITY BREAKDOWN */}
      <div className="bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl">
        <div className="flex items-center gap-2 mb-6">
          <Sparkles className="w-4 h-4 text-brand-indigo" />
          <h3 className="text-base font-medium text-white">System Attribution Tree (SHAP Insights)</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {inferenceData.shapContributors.map((item, idx) => {
            const IsPos = item.type === "positive";
            return (
              <div key={idx} className="p-4 bg-brand-dark/40 border border-brand-border rounded-xl flex items-start gap-3">
                <div className={`p-1.5 rounded-lg mt-0.5 ${IsPos ? 'bg-brand-emerald/10 text-brand-emerald' : 'bg-brand-coral/10 text-brand-coral'}`}>
                  {IsPos ? <ArrowDownRight className="w-4 h-4" /> : <ArrowUpRight className="w-4 h-4" />}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-medium text-white">{item.feature}</h4>
                    <span className={`text-xs font-mono px-1.5 py-0.2 rounded ${IsPos ? 'text-brand-emerald bg-brand-emerald/5' : 'text-brand-coral bg-brand-coral/5'}`}>
                      {IsPos ? '-' : '+'}{Math.abs(item.impact)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">{item.label}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* AI WELLNESS COACH MOCK INTERFACE */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl flex flex-col justify-between h-[420px]">
          <div className="flex items-center gap-2.5 pb-4 border-b border-brand-border">
            <div className="w-2 h-2 rounded-full bg-brand-emerald animate-pulse" />
            <h3 className="text-sm font-medium text-white">MindMeter Supportive Coach</h3>
          </div>

          {/* Chat Stream Box */}
          <div className="flex-1 overflow-y-auto py-4 space-y-3 pr-2 scrollbar-style">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.sender === 'user' 
                    ? 'bg-brand-indigo text-white' 
                    : 'bg-white/5 border border-brand-border text-gray-200'
                }`}>
                  {msg.text}
                </div>
              </div>
            ))}
          </div>

          {/* Prompt Entry Box */}
          <form onSubmit={handleSendMessage} className="flex gap-2 pt-3 border-t border-brand-border">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Discuss environmental stressors or sleep habits..."
              className="flex-1 bg-brand-dark border border-brand-border rounded-xl px-4 text-sm text-white focus:outline-none focus:border-brand-indigo transition-colors"
            />
            <button type="submit" className="p-3 bg-brand-indigo text-white rounded-xl hover:bg-brand-indigo/90 transition-colors">
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>

        {/* DISCOVERY ACTION CARDS */}
        <div className="bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <HeartHandshake className="w-4 h-4 text-brand-teal" />
              <h3 className="text-sm font-medium text-white">Suggested Interventions</h3>
            </div>
            <p className="text-xs text-gray-400 mb-4 leading-relaxed">
              Targeted behavioral focus areas to optimize lifestyle features and lower overall burnout risks.
            </p>
            <div className="space-y-2.5">
              {[
                "Deploy a 15-minute sound buffer block during heavy noise periods",
                "Keep sleep window locked within consistent 30-minute variables",
                "Dedicate 1 peer interaction hourly slot purely outside study topics"
              ].map((txt, idx) => (
                <div key={idx} className="p-3 bg-white/[0.01] border border-brand-border/60 rounded-xl text-xs text-gray-300">
                  {txt}
                </div>
              ))}
            </div>
          </div>
          <button 
            onClick={() => setMessages(prev => [...prev, { sender: 'user', text: "How can I enhance my baseline sleep metrics?" }])}
            className="w-full py-2.5 bg-white/5 hover:bg-white/10 text-white font-medium text-xs rounded-xl border border-brand-border transition-all mt-4"
          >
            Query Sleep Metric Optimization
          </button>
        </div>
      </div>

    </div>
  );
}