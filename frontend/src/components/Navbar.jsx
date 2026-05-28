import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, BarChart3, ClipboardCheck } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  
  const navItems = [
    { path: '/assessment', name: 'Check Metric', icon: ClipboardCheck },
    { path: '/results', name: 'My Core Dashboard', icon: LayoutDashboard },
    { path: '/analytics', name: 'Aggregated Trends', icon: BarChart3 },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-brand-dark/60 backdrop-blur-md border-b border-brand-border">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="p-2 bg-brand-indigo/10 rounded-xl group-hover:bg-brand-indigo/20 transition-all">
            <Activity className="w-5 h-5 text-brand-indigo" />
          </div>
          <span className="font-semibold tracking-tight text-white text-lg">MindMeter</span>
        </Link>
        
        <div className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-brand-border">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  isActive 
                    ? 'bg-brand-indigo text-white shadow-lg shadow-brand-indigo/20' 
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.name}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}