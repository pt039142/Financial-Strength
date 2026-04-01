import React from 'react';

export default function AboutPage() {
  return (
    <main className="flex min-h-screen flex-col items-center py-24 px-6 bg-dark text-white pt-32">
      <div className="container mx-auto max-w-4xl">
        <h1 className="text-5xl font-bold mb-12 text-gradient text-center">About Autonomiq.AI</h1>
        
        <div className="prose prose-invert max-w-none text-gray-400 space-y-8 text-lg">
          <p>
            Autonomiq.AI was founded to revolutionize financial operations through AI. We believe finance teams should focus on strategy, not manual data entry.
          </p>
          
          <div className="grid md:grid-cols-2 gap-12 py-12">
            <div>
              <h2 className="text-3xl font-bold text-white mb-4">Our Mission</h2>
              <p>To provide AI infrastructure that automates 90% of manual finance tasks, reducing human error and accelerating growth.</p>
            </div>
            <div>
              <h2 className="text-3xl font-bold text-white mb-4">Our Vision</h2>
              <p>A world where financial data flows autonomously, providing real-time insights and perfect accuracy for every company.</p>
            </div>
          </div>

          <h2 className="text-3xl font-bold text-white mt-12 mb-6">Why Autonomiq?</h2>
          <p>
            We leverage cutting-edge LLMs specifically trained on financial datasets. Our platform is built for scale, security, and precision.
          </p>
        </div>
      </div>
    </main>
  );
}