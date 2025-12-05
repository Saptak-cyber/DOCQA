import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Document Q&A System",
  description: "RAG-powered document question answering",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
