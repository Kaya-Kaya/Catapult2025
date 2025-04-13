'use client';

import React from 'react';
import Link from 'next/link';

const EasterEggGolfer = () => {
  // Placeholder for animation/image
  const golferGraphic = '🏌️'; // Simple emoji for now

  return (
    <Link
      href="/golf-game"
      className="fixed bottom-4 right-4 z-50 p-2 bg-green-200 rounded-full shadow-lg hover:bg-green-300 transition-colors cursor-pointer text-2xl animate-bounce"
      title="Play a game?"
    >
      {golferGraphic}
    </Link>
  );
};

export default EasterEggGolfer; 