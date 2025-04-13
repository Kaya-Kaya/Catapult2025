"use client";
import { useState, useRef } from 'react';
import Head from 'next/head';
import {GoogleGenAI} from '@google/genai';

const GEMINI_API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY;

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type === 'video/mp4') {
        setFile(selectedFile);
        const url = URL.createObjectURL(selectedFile);
        setPreview(url);
        setFeedback(null);
        setError(null);
      } else {
        setError('Please upload an MP4 video file only.');
        setFile(null);
        setPreview(null);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      if (droppedFile.type === 'video/mp4') {
        setFile(droppedFile);
        const url = URL.createObjectURL(droppedFile);
        setPreview(url);
        setFeedback(null);
        setError(null);
      } else {
        setError('Please upload an MP4 video file only.');
        setFile(null);
        setPreview(null);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    setError(null);
    setFeedback(null);
  
    try {
      const formData = new FormData();
      formData.append('video', file);

      /*const response = await fetch('/api/analyze-swing', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze swing');
      }

      const scores = data.scores;*/

      const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });
  
      const prompt = `
      You are a golfing expert who understands the theory of optimal golf swing mechanics.
      You are also a golf coach who can explain the theory to a beginner golfer.
      Do not be corny and be professional. Make your response 250 words max, but try not to write too little. Do not try to use quotation marks and any other text formatting (e.g. bolding or italics).
      Based on the following scores (each between 0.0 and 1.0), provide specific, meaningful, detailed, and constructive feedback on the golfer's swing in the following structured format (follow it exactly):
      - Posture: [Feedback on posture]
      - Swing Path: [Feedback on swing path]
      - Impact: [Feedback on impact]
      - Follow Through: [Feedback on follow-through]
      - Recommendations: [List of 3-5 specific recommendations to improve the swing]
  
      Scores:
      Ball Position: 1.0
      - Description: The position of the ball in relation to the golfer's stance.
      Drive Stance: 1.0
      - Description: The stance of the golfer when using a driver.
      Elbow Posture Backswing: 1.0
      - Description: The position of the elbows during the backswing.
      Elbow Posture Frontswing: 1.0
      - Description: The position of the elbows during the front swing.
      `;
  
      const result = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: prompt,
      });
      
      const feedbackText = result.text;
      const parsedFeedback = {
        posture: '',
        swingPath: '',
        impact: '',
        followThrough: '',
        recommendations: []
      };
      
      const lines = feedbackText.split('\n');
      let currentSection = '';
      
      for (const line of lines) {
        const trimmedLine = line.trim();
        
        if (trimmedLine.startsWith('- Posture:')) {
          currentSection = 'posture';
          parsedFeedback.posture = trimmedLine.replace('- Posture:', '').trim();
        } 
        else if (trimmedLine.startsWith('- Swing Path:')) {
          currentSection = 'swingPath';
          parsedFeedback.swingPath = trimmedLine.replace('- Swing Path:', '').trim();
        } 
        else if (trimmedLine.startsWith('- Impact:')) {
          currentSection = 'impact';
          parsedFeedback.impact = trimmedLine.replace('- Impact:', '').trim();
        } 
        else if (trimmedLine.startsWith('- Follow Through:')) {
          currentSection = 'followThrough';
          parsedFeedback.followThrough = trimmedLine.replace('- Follow Through:', '').trim();
        } 
        else if (trimmedLine.startsWith('- Recommendations:')) {
          currentSection = 'recommendations';
        } 
        else if (currentSection === 'recommendations' && /^\d+\./.test(trimmedLine)) {
          // Match numbered recommendations (1., 2., etc.)
          parsedFeedback.recommendations.push(trimmedLine.replace(/^\d+\./, '').trim());
        }
        else if (currentSection && currentSection !== 'recommendations' && trimmedLine) {
          // Append continuation text to current section (except recommendations)
          parsedFeedback[currentSection] += ' ' + trimmedLine;
        }
      }
      
      // Clean up each section by removing extra spaces
      Object.keys(parsedFeedback).forEach(key => {
        if (key !== 'recommendations') {
          parsedFeedback[key] = parsedFeedback[key].replace(/\s+/g, ' ').trim();
        }
      });
      
      setFeedback(parsedFeedback);
    } catch (err) {
      console.error('Error analyzing swing:', err);
      setError('There was an error analyzing your swing. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <Head>
        <title>Golfmate - AI Golf Swing Analysis</title>
        <meta name="description" content="Improve your golf swing with AI-powered analysis" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="container mx-auto px-4 py-6">
        <nav className="flex justify-between items-center">
          <div className="flex items-center">
            <svg className="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12m-2.879 2.879a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z" />
            </svg>
            <h1 className="ml-2 text-2xl font-bold text-gray-800">Golfmate</h1>
          </div>
        </nav>
      </header>

      <main className="container mx-auto px-4 py-12">
        <section className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">Perfect Your Swing with AI</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload your golf swing video and get instant, professional-level analysis.
          </p>
        </section>

        <section className="max-w-5xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="md:flex">
            <div className="md:w-1/2 p-8">
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">Upload Your Swing</h3>
              {error && (
                <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-md">
                  {error}
                </div>
              )}
              <div
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:bg-gray-50 transition"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
              >
                <input
                  type="file"
                  accept="video/mp4"
                  className="hidden"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                />
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-2 text-sm text-gray-600">Drag and drop your MP4 video here, or click to browse</p>
                <p className="mt-1 text-xs text-gray-500">MP4 format only, up to 100MB</p>
              </div>

              {preview && (
                <div className="mt-6">
                  <video 
                    src={preview} 
                    controls 
                    className="w-full h-auto rounded-lg"
                  />
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className={`mt-4 w-full py-3 px-4 rounded-md font-medium text-white ${
                      isAnalyzing ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'
                    } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition cursor-pointer`}
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze My Swing'}
                  </button>
                </div>
              )}
            </div>

            <div className="md:w-1/2 bg-gray-50 p-8">
              <h3 className="text-2xl font-semibold text-gray-800 mb-4">Feedback</h3>
              
              {!feedback && !isAnalyzing && (
                <div className="h-64 flex items-center justify-center">
                  <p className="text-gray-500 text-center">Upload your swing to see personalized feedback</p>
                </div>
              )}
              
              {isAnalyzing && (
                <div className="h-64 flex flex-col items-center justify-center">
                  <svg className="animate-spin h-10 w-10 text-green-600 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <p className="text-gray-600">Analyzing your golf swing...</p>
                </div>
              )}
              
              {feedback && (
                <div className="space-y-4">
                  {feedback.posture && (
                    <div>
                      <h4 className="font-bold text-gray-700">Posture</h4>
                      <p className="text-gray-600 whitespace-pre-line">{feedback.posture}</p>
                    </div>
                  )}
                  
                  {feedback.swingPath && (
                    <div>
                      <h4 className="font-bold text-gray-700">Swing Path</h4>
                      <p className="text-gray-600 whitespace-pre-line">{feedback.swingPath}</p>
                    </div>
                  )}
                  
                  {feedback.impact && (
                    <div>
                      <h4 className="font-bold text-gray-700">Impact</h4>
                      <p className="text-gray-600 whitespace-pre-line">{feedback.impact}</p>
                    </div>
                  )}
                  
                  {feedback.followThrough && (
                    <div>
                      <h4 className="font-bold text-gray-700">Follow Through</h4>
                      <p className="text-gray-600 whitespace-pre-line">{feedback.followThrough}</p>
                    </div>
                  )}
                  
                  <div>
                    <h4 className="font-bold text-gray-700">Recommendations</h4>
                    {feedback.recommendations && feedback.recommendations.length > 0 ? (
                      <ol className="list-decimal pl-5 text-gray-600 space-y-2">
                        {feedback.recommendations.map((rec, index) => (
                          <li key={index} className="whitespace-pre-line">{rec}</li>
                        ))}
                      </ol>
                    ) : (
                      <p className="text-gray-600">No specific recommendations provided.</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-gray-800 text-white mt-20 py-6">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400">© {new Date().getFullYear()} Golfmate. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}