import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import EasterEggGolfer from "../components/EasterEggGolfer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "Golfmate",
  description: "Catapult 2025",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
        <EasterEggGolfer />
      </body>
    </html>
  );
}
