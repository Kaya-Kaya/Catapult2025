# GolfMate

## Introduction

GolfMate is a web application designed to help golfers improve their swing using AI-powered analysis. Upload a video of your swing, and GolfMate provides constructive feedback and recommendations. 

## Description

The core idea of GolfMate is to provide golfers with accessible, data-driven feedback on their swing mechanics. The planned workflow involves:

1.  **Video Upload**: Users upload an MP4 video of their golf swing.
2.  **Frame Extraction & Analysis**: The video is processed to extract frames and identify key posture nodes (e.g., joints, club position).
3.  **Neural Network Scoring**: A neural network analyzes the posture node coordinates across frames to generate scores for different aspects of the swing (e.g., posture, swing path, impact). 
4.  **Feedback Generation**: An LLM (currently Google Gemini) takes the scores and generates structured, constructive feedback, including specific recommendations for improvement.

## Features

*   **Video Upload**: Supports MP4 video uploads via drag-and-drop or file browser.
*   **Video Preview**: Shows a preview of the uploaded video.
*   **AI Swing Analysis**: Analyzes the swing (currently using Google Gemini API with a static prompt) and provides feedback on 9 key metrics.
*   **Personalized Recommendations**: Offers specific suggestions to improve the swing.

**Project Structure:**

```
golfmate/
├── public/             # Static assets
├── src/
│   ├── app/            # Next.js App Router
│   │   ├── page.js     # Main swing analysis page component
│   │   ├── layout.js   # Root layout
│   │   ├── globals.css # Global styles
│   │   └── golf-game/
│   │       └── page.tsx # 2D Mini Golf game component
│   └── components/
│       └── EasterEggGolfer.tsx # Link component to the golf game
├── next.config.mjs     # Next.js configuration
├── tailwind.config.js  # Tailwind CSS configuration
├── package.json        # Project dependencies and scripts
└── README.md           # This file
```

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd golfmate 
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    # or
    bun install
    ```

3.  **Set up environment variables:**
    Create a `.env.local` file in the `golfmate` directory and add your Google Gemini API key:
    ```
    NEXT_PUBLIC_GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    ```

4.  **Run the development server:**
    ```bash
    npm run dev
    # or
    bun dev
    ```

5.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.
    *   The main swing analysis page is at the root (`/`).
    *   The mini-golf game is at `/golf-game`.