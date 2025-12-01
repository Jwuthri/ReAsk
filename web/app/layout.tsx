import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ReAsk - LLM Conversation Evaluator',
  description: 'Detect bad LLM responses through re-ask detection analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

