import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, BookOpen, Users, Trees, Dumbbell, 
  ArrowLeft, ArrowRight, CheckCircle, FlameKindling 
} from 'lucide-react';

// Organized inputs mapped strictly to the dataset features
const CATEGORIES = [
  {
    id: 'psychological',
    title: 'Psychological Factors',
    icon: Brain,
    fields: [
      { id: 'self_esteem', label: 'Self Esteem Level', min: 0, max: 30, desc: 'Perceived self worth threshold score' },
      { id: 'mental_health_history', label: 'Mental Health History Indicator', min: 0, max: 1, desc: '0: No recorded historical issues, 1: History exists' },
    ]
  },
  {
    id: 'academic',
    title: 'Academic Metrics',
    icon: BookOpen,
    fields: [
      { id: 'academic_performance', label: 'Academic Performance', min: 0, max: 5, desc: 'Current generalized index output' },
      { id: 'study_load', label: 'Perceived Study Load', min: 0, max: 5, desc: 'Assessed scale of core volume intensity' },
      { id: 'teacher_student_relationship', label: 'Teacher-Student Relationship Status', min: 0, max: 5, desc: 'Communication quality index' },
      { id: 'future_career_concerns', label: 'Future Career Concerns Weight', min: 0, max: 5, desc: 'Anxiety intensity concerning employment matching' },
    ]
  },
  {
    id: 'social',
    title: 'Social Integrations',
    icon: Users,
    fields: [
      { id: 'social_support', label: 'Social Support Access', min: 0, max: 3, desc: 'Availability tier of close peer circles' },
      { id: 'peer_pressure', label: 'Peer Pressure Index', min: 0, max: 5, desc: 'Friction density felt from societal environments' },
      { id: 'bullying', label: 'Bullying Frequency Exposure', min: 0, max: 5, desc: 'Encounter frequency matrix metric' },
    ]
  },
  {
    id: 'environment',
    title: 'Environmental Conditions',
    icon: Trees,
    fields: [
      { id: 'noise_level', label: 'Surrounding Noise Level Exposure', min: 0, max: 5, desc: 'Acoustic background disruption index' },
      { id: 'living_conditions', label: 'Living Conditions Grade', min: 0, max: 5, desc: 'Basic safety/comfort density index' },
      { id: 'safety', label: 'Perceived General Safety Status', min: 0, max: 5, desc: 'Environmental threat absence calibration' },
      { id: 'basic_needs', label: 'Basic Needs Fulfillment Rate', min: 0, max: 5, desc: 'Dietary, baseline security and logistical comfort status' },
    ]
  },
  {
    id: 'lifestyle',
    title: 'Physiological & Lifestyle',
    icon: Dumbbell,
    fields: [
      { id: 'sleep_quality', label: 'Sleep Quality Value', min: 0, max: 5, desc: 'Rest efficiency and rhythm integrity' },
      { id: 'headache', label: 'Headache Frequency Index', min: 0, max: 5, desc: 'Somatic distress manifestation indicator' },
      { id: 'breathing_problem', label: 'Breathing Problem Tendency', min: 0, max: 5, desc: 'Somatic feedback loop disruption points' },
      { id: 'extracurricular_activities', label: 'Extracurricular Engagement', min: 0, max: 5, desc: 'Active deliberate decompression allocation' },
    ]
  }
];

// Initialize clean initial payload values
const INITIAL_FORM_STATE = {
  self_esteem: 15, mental_health_history: 0, headache: 1, sleep_quality: 4,
  breathing_problem: 0, noise_level: 2, living_conditions: 4, safety: 4,
  basic_needs: 4, academic_performance: 4, study_load: 2, teacher_student_relationship: 4,
  future_career_concerns: 2, social_support: 3, peer_pressure: 1, extracurricular_activities: 3,
  bullying: 0
};

export default function AssessmentPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState(INITIAL_FORM_STATE);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const navigate = useNavigate();
  const currentCategory = CATEGORIES[currentStep];
  const StepIcon = currentCategory.icon;

  const handleSliderChange = (fieldId, value) => {
    setFormData(prev => ({ ...prev, [fieldId]: Number(value) }));
  };

  const handleNext = () => {
    if (currentStep < CATEGORIES.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      handleSubmit();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) setCurrentStep(prev => prev - 1);
  };

  const handleSubmit = () => {
    setIsSubmitting(true);
    // Simulating quick machine learning runtime classification validation
    setTimeout(() => {
      setIsSubmitting(false);
      // Pass states onto the results platform view
      navigate('/results', { state: { rawInputs: formData } });
    }, 1200);
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-12 flex-1 flex flex-col justify-center w-full">
      {/* Gamified Node Map Tracking Progress */}
      <div className="flex items-center justify-between mb-10 bg-brand-card border border-brand-border p-4 rounded-xl">
        {CATEGORIES.map((cat, idx) => {
          const Icon = cat.icon;
          const isCompleted = idx < currentStep;
          const isCurrent = idx === currentStep;
          return (
            <div key={cat.id} className="flex items-center flex-1 last:flex-none">
              <div className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all ${
                isCurrent ? 'bg-brand-indigo text-white ring-4 ring-brand-indigo/10' :
                isCompleted ? 'bg-brand-emerald/20 text-brand-emerald' : 'bg-white/5 text-gray-500'
              }`}>
                {isCompleted ? <CheckCircle className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
              </div>
              {idx < CATEGORIES.length - 1 && (
                <div className={`h-0.5 flex-1 mx-2 rounded-full ${idx < currentStep ? 'bg-brand-emerald/40' : 'bg-brand-border'}`} />
              )}
            </div>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 15 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -15 }}
          transition={{ duration: 0.3 }}
          className="bg-brand-card border border-brand-border rounded-2xl p-6 md:p-8 backdrop-blur-md shadow-2xl relative"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 bg-brand-indigo/10 rounded-xl text-brand-lavender">
              <StepIcon className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-brand-indigo">System Partition {currentStep + 1} of 5</span>
              <h2 className="text-xl font-medium text-white">{currentCategory.title}</h2>
            </div>
          </div>

          <div className="space-y-6">
            {currentCategory.fields.map((field) => (
              <div key={field.id} className="p-4 bg-white/[0.02] border border-brand-border/60 rounded-xl hover:border-brand-border transition-colors">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-sm font-medium text-gray-200">{field.label}</label>
                  <span className="text-base font-bold text-brand-teal bg-brand-teal/10 px-2.5 py-0.5 rounded-md min-w-[2.5rem] text-center">
                    {formData[field.id]}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mb-4">{field.desc}</p>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-gray-500 font-mono">{field.min}</span>
                  <input
                    type="range"
                    min={field.min}
                    max={field.max}
                    step={1}
                    value={formData[field.id]}
                    onChange={(e) => handleSliderChange(field.id, e.target.value)}
                    className="flex-1 h-1.5 bg-brand-dark rounded-lg appearance-none cursor-pointer accent-brand-indigo focus:outline-none"
                  />
                  <span className="text-xs text-gray-500 font-mono">{field.max}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between mt-8 pt-6 border-t border-brand-border">
            <button
              onClick={handleBack}
              disabled={currentStep === 0}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                currentStep === 0 ? 'text-gray-600 cursor-not-allowed' : 'text-gray-300 hover:text-white hover:bg-white/5'
              }`}
            >
              <ArrowLeft className="w-4 h-4" />
              Previous
            </button>

            <button
              onClick={handleNext}
              disabled={isSubmitting}
              className="flex items-center gap-2 px-6 py-3 bg-brand-indigo hover:bg-brand-indigo/90 text-white text-sm font-medium rounded-xl shadow-lg shadow-brand-indigo/10 transition-all ml-auto"
            >
              {isSubmitting ? (
                <>
                  <FlameKindling className="w-4 h-4 animate-spin text-brand-teal" />
                  Running Calibration Model...
                </>
              ) : currentStep === CATEGORIES.length - 1 ? (
                <>
                  Compute Burnout Vector
                  <CheckCircle className="w-4 h-4" />
                </>
              ) : (
                <>
                  Next Metric Section
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}