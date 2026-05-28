import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';

export default function SharedLayout() {
  return (
    <div className="min-h-screen bg-brand-dark text-gray-100 flex flex-col relative selection:bg-brand-indigo selection:text-white">
      {/* Background Soft Gradients */}
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-brand-indigo/10 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute bottom-10 right-1/4 w-[600px] h-[600px] bg-brand-teal/5 rounded-full filter blur-[150px] pointer-events-none" />
      
      <Navbar />
      <main className="flex-1 pt-16 flex flex-col">
        <Outlet />
      </main>
    </div>
  );
}