import React from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, AreaChart, Area } from 'recharts';
import { Layers, TrendingUp, HelpCircle } from 'lucide-react';

export default function AnalyticsPage() {
  // Aggregate macro data structures matching the XGBoost training setup
  const distributionData = [
    { name: 'Healthy', volume: 420 },
    { name: 'Moderate', volume: 310 },
    { name: 'High', volume: 185 },
    { name: 'Critical', volume: 65 },
  ];

  const globalFeatureImportance = [
    { name: 'Study Load', weight: 88 },
    { name: 'Sleep Quality', weight: 82 },
    { name: 'Peer Pressure', weight: 74 },
    { name: 'Self Esteem', weight: 65 },
    { name: 'Social Support', weight: 58 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-10 space-y-8 w-full">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Aggregated Platform Intelligence</h1>
        <p className="text-gray-400 text-sm">Macro-level structural insights and performance indexes across cohorts.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* COHORT RISK DISTRIBUTION CHART */}
        <div className="bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <Layers className="w-4 h-4 text-brand-indigo" />
            <h3 className="text-sm font-medium text-white">Risk Target Category Density</h3>
          </div>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" stroke="#6B7280" fontSize={11} tickLine={false} />
                <YAxis stroke="#6B7280" fontSize={11} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#161A29', borderColor: 'rgba(255,255,255,0.06)', borderRadius: '12px' }}
                  itemStyle={{ color: '#F3F4F6', fontSize: '12px' }}
                />
                <Bar dataKey="volume" fill="#6366F1" radius={[6, 6, 0, 0]} maxBarSize={45} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* MODEL COMPONENT WEIGHT INDEX */}
        <div className="bg-brand-card border border-brand-border rounded-2xl p-6 shadow-xl">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="w-4 h-4 text-brand-teal" />
            <h3 className="text-sm font-medium text-white">Global Feature Importance (XGBoost F-Score)</h3>
          </div>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={globalFeatureImportance} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" stroke="#6B7280" fontSize={11} tickLine={false} />
                <YAxis stroke="#6B7280" fontSize={11} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#161A29', borderColor: 'rgba(255,255,255,0.06)', borderRadius: '12px' }}
                  itemStyle={{ color: '#F3F4F6', fontSize: '12px' }}
                />
                <Area type="monotone" dataKey="weight" stroke="#14B8A6" fill="rgba(20, 184, 166, 0.1)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* REASSURING TECHNICAL METRIC NOTES */}
      <div className="bg-brand-card border border-brand-border rounded-2xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-white/5 rounded-xl text-brand-lavender">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-sm font-medium text-white">How to interpret these platform analytics?</h4>
            <p className="text-xs text-gray-400 mt-0.5 max-w-xl">
              These tracking distributions show structural weights across thousands of anonymous input points. High study loads are normal during exam clusters; our platform adjusts to differentiate temporary crunch from genuine structural risk.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}