import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, Sparkles, Zap, ArrowRight, HeartPulse } from 'lucide-react';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-6 relative overflow-hidden py-12">
      {/* Dynamic Animated Particles */}
      <div className="absolute inset-0 pointer-events-none opacity-40">
        {[...Array(12)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1.5 h-1.5 bg-brand-lavender rounded-full"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -40, 0],
              opacity: [0.2, 0.8, 0.2],
            }}
            transition={{
              duration: 5 + Math.random() * 5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      <div className="max-w-4xl text-center z-10">
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-indigo/10 border border-brand-indigo/20 text-brand-lavender text-xs font-medium mb-6"
        >
          <Sparkles className="w-3.5 h-3.5" />
          Predictive Academic Burnout Intelligence
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-5xl md:text-6xl font-bold tracking-tight text-white mb-6 leading-[1.15]"
        >
          Understand Burnout <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-brand-indigo via-brand-lavender to-brand-teal">
            Before It Understands You.
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto mb-10 font-normal leading-relaxed"
        >
          AI-powered student wellbeing intelligence designed to identify burnout risk and provide supportive guidance.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20"
        >
          <button
            onClick={() => navigate('/assessment')}
            className="w-full sm:w-auto px-8 py-4 bg-brand-indigo hover:bg-brand-indigo/90 text-white font-medium rounded-xl shadow-xl shadow-brand-indigo/20 transition-all flex items-center justify-center gap-2 group"
          >
            Start Dynamic Assessment
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
          <button
            onClick={() => navigate('/analytics')}
            className="w-full sm:w-auto px-8 py-4 bg-white/5 hover:bg-white/10 text-white font-medium rounded-xl border border-brand-border transition-all"
          >
            Explore Metric Analytics
          </button>
        </motion.div>
      </div>

      {/* Gamified Core System Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl w-full z-10 border-t border-brand-border/40 pt-16">
        {[
          {
            icon: Shield,
            title: "Leakage-Free Engine",
            desc: "Predicts state natively through lifestyle signals, tracking patterns without looking at direct clinical diagnoses.",
          },
          {
            icon: HeartPulse,
            title: "Supportive Posture",
            desc: "Zero judgment built-in. MindMeter functions strictly as a peer-level helper optimized to foster focus recovery.",
          },
          {
            icon: Zap,
            title: "SHAP Explainability",
            desc: "Transparent model tracking explains exactly why indicators spike, preventing black-box uncertainty.",
          },
        ].map((feat, idx) => {
          const Icon = feat.icon;
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              className="p-6 rounded-2xl bg-brand-card border border-brand-border backdrop-blur-sm hover:border-brand-indigo/30 transition-all group"
            >
              <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center mb-4 group-hover:bg-brand-indigo/10 transition-colors">
                <Icon className="w-5 h-5 text-brand-indigo" />
              </div>
              <h3 className="text-white font-medium text-base mb-2">{feat.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{feat.desc}</p>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}