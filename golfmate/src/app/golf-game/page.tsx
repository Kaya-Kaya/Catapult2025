'use client';

import React, { useEffect, useRef, useState } from 'react';
import Link from 'next/link';

const GolfGamePage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [gameState, setGameState] = useState<'idle' | 'aiming' | 'ballMoving' | 'finished'>('idle');
  const [score, setScore] = useState<number | null>(null);
  const [message, setMessage] = useState<string>('Click Start Game to begin');
  
  // Game state refs (used in the animation loop)
  const gameStateRef = useRef(gameState);
  const ballRef = useRef({ x: 0, y: 0, vx: 0, vy: 0, radius: 6 });
  const holeRef = useRef({ x: 0, y: 0, radius: 9 });
  const aimRef = useRef({ active: false, startX: 0, startY: 0, endX: 0, endY: 0 });
  const requestRef = useRef<number | undefined>(undefined);
  
  // Course elements
  const sandTrapRef = useRef({ x: 0, y: 0, width: 0, height: 0 });
  
  // Constants
  const FRICTION = 0.98;
  const SAND_FRICTION = 0.92;
  const MIN_VELOCITY = 0.1;
  const MAX_POWER = 15;
  
  useEffect(() => {
    gameStateRef.current = gameState;
  }, [gameState]);
  
  useEffect(() => {
    // Initialize the canvas when component mounts
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Setup canvas dimensions based on container
    const resizeCanvas = () => {
      const container = canvas.parentElement;
      if (container) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        
        // Reset ball and hole positions based on new dimensions
        resetGame();
      }
    };
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, []);
  
  // Reset game state
  const resetGame = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Position ball near the bottom center
    ballRef.current = {
      x: canvas.width / 2,
      y: canvas.height - 100,
      vx: 0,
      vy: 0,
      radius: 6
    };
    
    // Position hole near top of canvas
    holeRef.current = {
      x: canvas.width / 2,
      y: 80,
      radius: 9
    };
    
    // Position sand trap
    sandTrapRef.current = {
      x: canvas.width / 3,
      y: canvas.height / 2,
      width: canvas.width / 4,
      height: canvas.height / 6
    };
    
    aimRef.current = { active: false, startX: 0, startY: 0, endX: 0, endY: 0 };
    
    if (gameStateRef.current === 'idle') {
      setMessage('Click Start Game to begin');
    } else {
      setMessage('Click and drag to aim, release to shoot');
    }
    
    setScore(null);
  };
  
  // Start the game
  const startGame = () => {
    setGameState('aiming');
    resetGame();
    startGameLoop();
  };
  
  // Animation loop
  const gameLoop = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw golf course (green background with some details)
    drawCourse(ctx, canvas.width, canvas.height);
    
    // Draw hole
    drawHole(ctx);
    
    // Draw ball
    drawBall(ctx);
    
    // If aiming, draw aim line
    if (gameStateRef.current === 'aiming' && aimRef.current.active) {
      drawAimLine(ctx);
    }
    
    // Ball physics when moving
    if (gameStateRef.current === 'ballMoving') {
      // Apply physics
      const ball = ballRef.current;
      
      // Check if ball is in sand trap
      const sandTrap = sandTrapRef.current;
      const inSandTrap = (
        ball.x > sandTrap.x && 
        ball.x < sandTrap.x + sandTrap.width && 
        ball.y > sandTrap.y && 
        ball.y < sandTrap.y + sandTrap.height
      );
      
      // Apply position updates
      ball.x += ball.vx;
      ball.y += ball.vy;
      
      // Apply appropriate friction
      if (inSandTrap) {
        ball.vx *= SAND_FRICTION;
        ball.vy *= SAND_FRICTION;
      } else {
        ball.vx *= FRICTION;
        ball.vy *= FRICTION;
      }
      
      // Check if ball is in hole (before checking if stopped)
      const hole = holeRef.current;
      const distanceToHole = Math.sqrt(
        Math.pow(ball.x - hole.x, 2) + Math.pow(ball.y - hole.y, 2)
      );
      
      if (distanceToHole < hole.radius) {
        // Ball is in hole - stop ball movement and end game
        ball.vx = 0;
        ball.vy = 0;
        setGameState('finished');
        setScore(100);
        setMessage('Swing so fake it might\'ve voted twice.');
        return; // Exit game loop early
      }
      
      // Check if ball stopped
      if (Math.abs(ball.vx) < MIN_VELOCITY && Math.abs(ball.vy) < MIN_VELOCITY) {
        ball.vx = 0;
        ball.vy = 0;
        
        // Ball stopped but not in hole
        setGameState('aiming');
        const distanceFromHole = Math.round(distanceToHole);
        const calculatedScore = Math.max(0, Math.round(100 - distanceToHole / 5));
        setScore(calculatedScore);
        setMessage(`Ball stopped ${distanceFromHole}px from hole. Try again!`);
      }
      
      // Check for wall collisions
      if (ball.x < ball.radius || ball.x > canvas.width - ball.radius) {
        ball.vx *= -0.7; // Bounce with energy loss
        ball.x = ball.x < ball.radius ? ball.radius : canvas.width - ball.radius;
      }
      
      if (ball.y < ball.radius || ball.y > canvas.height - ball.radius) {
        ball.vy *= -0.7; // Bounce with energy loss
        ball.y = ball.y < ball.radius ? ball.radius : canvas.height - ball.radius;
      }
    }
    
    // Display score or messages
    drawUI(ctx, canvas.width, canvas.height);
    
    // Continue animation loop
    requestRef.current = requestAnimationFrame(gameLoop);
  };
  
  const startGameLoop = () => {
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    requestRef.current = requestAnimationFrame(gameLoop);
  };
  
  // Drawing functions
  const drawCourse = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Background
    ctx.fillStyle = '#111927';
    ctx.fillRect(0, 0, width, height);
    
    // Define control points for cloud-shaped course
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Draw amoeba/cloud-shaped golf course
    ctx.fillStyle = '#4CBB17';
    ctx.beginPath();
    
    // Starting point
    ctx.moveTo(centerX - width * 0.4, centerY);
    
    // Draw cloud/amoeba blob with multiple curves
    // Left side
    ctx.bezierCurveTo(
      centerX - width * 0.5, centerY - height * 0.2, // control point 1
      centerX - width * 0.3, centerY - height * 0.3, // control point 2
      centerX - width * 0.1, centerY - height * 0.25 // end point
    );
    
    // Top bulge
    ctx.bezierCurveTo(
      centerX, centerY - height * 0.4, // control point 1
      centerX + width * 0.15, centerY - height * 0.4, // control point 2
      centerX + width * 0.2, centerY - height * 0.25 // end point
    );
    
    // Right top bulge
    ctx.bezierCurveTo(
      centerX + width * 0.3, centerY - height * 0.2, // control point 1
      centerX + width * 0.4, centerY - height * 0.15, // control point 2
      centerX + width * 0.35, centerY // end point
    );
    
    // Right bottom bulge
    ctx.bezierCurveTo(
      centerX + width * 0.4, centerY + height * 0.2, // control point 1
      centerX + width * 0.3, centerY + height * 0.3, // control point 2
      centerX + width * 0.1, centerY + height * 0.25 // end point
    );
    
    // Bottom bulge
    ctx.bezierCurveTo(
      centerX, centerY + height * 0.3, // control point 1
      centerX - width * 0.3, centerY + height * 0.3, // control point 2
      centerX - width * 0.4, centerY // end point
    );
    
    ctx.closePath();
    ctx.fill();
    
    // Draw a lighter shade for the "fairway" path within the cloud shape
    ctx.fillStyle = '#7CFC00';
    ctx.beginPath();
    
    // Starting point for fairway (near ball position)
    const fairwayStartX = width / 2;
    const fairwayStartY = height - 100;
    
    ctx.moveTo(fairwayStartX - 30, fairwayStartY);
    
    // Curved fairway path to the hole
    ctx.bezierCurveTo(
      fairwayStartX - 50, centerY + height * 0.1, // control point 1
      fairwayStartX + 50, centerY - height * 0.1, // control point 2
      centerX, holeRef.current.y + 20 // end near hole
    );
    
    // Widen the path
    ctx.bezierCurveTo(
      fairwayStartX + 100, centerY - height * 0.1, // control point 1
      fairwayStartX + 50, centerY + height * 0.1, // control point 2
      fairwayStartX + 30, fairwayStartY // back to start, but wider
    );
    
    ctx.closePath();
    ctx.fill();
    
    // Draw sand trap
    ctx.fillStyle = '#F4D03F'; // Sand color
    ctx.beginPath();
    const sandTrap = sandTrapRef.current;
    ctx.ellipse(
      sandTrap.x + sandTrap.width/2, 
      sandTrap.y + sandTrap.height/2, 
      sandTrap.width/2, 
      sandTrap.height/2, 
      0, 0, Math.PI * 2
    );
    ctx.fill();
    
    // Draw rough texture around the edges of the cloud
    ctx.strokeStyle = '#3a9012';
    ctx.lineWidth = 3;
    ctx.stroke();
  };
  
  const drawHole = (ctx: CanvasRenderingContext2D) => {
    const hole = holeRef.current;
    
    // Draw hole shadow
    ctx.beginPath();
    ctx.arc(hole.x, hole.y, hole.radius + 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fill();
    
    // Draw hole
    ctx.beginPath();
    ctx.arc(hole.x, hole.y, hole.radius, 0, Math.PI * 2);
    ctx.fillStyle = '#111';
    ctx.fill();
    
    // Draw flag pole
    ctx.beginPath();
    ctx.moveTo(hole.x, hole.y);
    ctx.lineTo(hole.x, hole.y - 40);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw flag
    ctx.beginPath();
    ctx.moveTo(hole.x, hole.y - 40);
    ctx.lineTo(hole.x + 15, hole.y - 35);
    ctx.lineTo(hole.x, hole.y - 30);
    ctx.fillStyle = 'red';
    ctx.fill();
  };
  
  const drawBall = (ctx: CanvasRenderingContext2D) => {
    const ball = ballRef.current;
    
    // Draw ball shadow
    ctx.beginPath();
    ctx.arc(ball.x + 2, ball.y + 2, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fill();
    
    // Draw ball
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.stroke();
  };
  
  const drawAimLine = (ctx: CanvasRenderingContext2D) => {
    const { startX, startY, endX, endY } = aimRef.current;
    
    // Calculate power based on drag distance
    const dragDistance = Math.sqrt(
      Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2)
    );
    const powerRatio = Math.min(dragDistance / 100, 1);
    
    // Draw aim line (direction is from start to end)
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = `rgba(255, ${Math.round(255 * (1 - powerRatio))}, 0, 0.8)`;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw power meter
    const powerMeterWidth = 50;
    const powerFill = powerRatio * powerMeterWidth;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(startX - 25, startY + 15, powerMeterWidth, 6);
    
    ctx.fillStyle = `rgb(255, ${Math.round(255 * (1 - powerRatio))}, 0)`;
    ctx.fillRect(startX - 25, startY + 15, powerFill, 6);
  };
  
  const drawUI = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Draw message
    ctx.font = '18px Arial';
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 4;
    ctx.strokeText(message, width / 2, 30);
    ctx.fillText(message, width / 2, 30);
    
    // Draw score if available
    if (score !== null) {
      ctx.font = '16px Arial';
      ctx.fillStyle = score > 70 ? 'lightgreen' : score > 40 ? 'yellow' : 'orange';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 3;
      ctx.strokeText(`Score: ${score}`, width / 2, 60);
      ctx.fillText(`Score: ${score}`, width / 2, 60);
    }
  };
  
  // Helper function to draw a cloud
  const drawCloud = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number) => {
    ctx.beginPath();
    ctx.arc(x, y, size * 0.5, 0, Math.PI * 2);
    ctx.arc(x + size * 0.4, y - size * 0.1, size * 0.4, 0, Math.PI * 2);
    ctx.arc(x + size * 0.8, y, size * 0.5, 0, Math.PI * 2);
    ctx.arc(x + size * 0.4, y + size * 0.1, size * 0.4, 0, Math.PI * 2);
    ctx.fill();
  };
  
  // Handle mouse/touch events
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (gameStateRef.current !== 'aiming') return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if click is near ball
    const ball = ballRef.current;
    const distance = Math.sqrt(Math.pow(x - ball.x, 2) + Math.pow(y - ball.y, 2));
    
    if (distance < ball.radius * 3) {
      aimRef.current = {
        active: true,
        startX: ball.x,
        startY: ball.y,
        endX: x,
        endY: y
      };
    }
  };
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!aimRef.current.active) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    aimRef.current.endX = e.clientX - rect.left;
    aimRef.current.endY = e.clientY - rect.top;
  };
  
  const handleMouseUp = () => {
    if (!aimRef.current.active) return;
    
    // Calculate velocity based on aim distance and direction
    const { startX, startY, endX, endY } = aimRef.current;
    
    // Direction is from ball to start point (opposite of visual aim line)
    // This makes it feel like pulling back and releasing
    const dx = startX - endX;
    const dy = startY - endY;
    
    // Calculate power based on drag distance
    const dragDistance = Math.sqrt(dx * dx + dy * dy);
    const power = Math.min(dragDistance / 10, MAX_POWER);
    
    // Normalize direction and apply power
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance > 0) {
      ballRef.current.vx = (dx / distance) * power;
      ballRef.current.vy = (dy / distance) * power;
      setGameState('ballMoving');
      setMessage('Ball is moving...');
    }
    
    aimRef.current.active = false;
  };
  
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (gameStateRef.current !== 'aiming') return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    
    // Check if touch is near ball
    const ball = ballRef.current;
    const distance = Math.sqrt(Math.pow(x - ball.x, 2) + Math.pow(y - ball.y, 2));
    
    if (distance < ball.radius * 3) {
      aimRef.current = {
        active: true,
        startX: ball.x,
        startY: ball.y,
        endX: x,
        endY: y
      };
    }
  };
  
  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!aimRef.current.active) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    aimRef.current.endX = touch.clientX - rect.left;
    aimRef.current.endY = touch.clientY - rect.top;
  };
  
  const handleTouchEnd = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    handleMouseUp();
  };
  
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-blue-300 to-green-300 p-4">
      <div className="w-full max-w-4xl">
        <div className="flex justify-between items-center mb-4">
          <Link href="/" className="text-sm text-blue-100 hover:text-white bg-black/30 px-3 py-1 rounded">
            &larr; Back to Home
          </Link>
          
          {gameState === 'idle' ? (
            <button
              onClick={startGame}
              className="px-6 py-3 bg-green-500 text-white text-xl font-bold rounded shadow-lg hover:bg-green-600 transition-colors"
            >
              Start Game
            </button>
          ) : (
            <button
              onClick={() => setGameState('idle')}
              className="px-4 py-2 bg-red-500 text-white font-semibold rounded shadow hover:bg-red-600 transition-colors"
            >
              Exit Game
            </button>
          )}
        </div>
        
        <h1 className="text-4xl font-bold text-center text-white drop-shadow-md mb-4">
          Cloud Shape Golf
        </h1>
        
        <div className="bg-white rounded-lg shadow-xl overflow-hidden border-4 border-white aspect-video">
          <canvas
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            className="w-full h-full cursor-crosshair"
          />
        </div>
        
        <div className="mt-4 text-center text-white">
          <p>Aim by clicking near the ball and dragging backward, then release to shoot</p>
          {gameState === 'finished' && score === 100 && (
            <div className="mt-2 p-4 bg-green-700 rounded-lg inline-block text-lg font-bold">
              Swing so fake it might've voted twice.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GolfGamePage;