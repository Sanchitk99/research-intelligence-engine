import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Research Intelligence Engine | Evidence-Grounded AI',
  description: 'Production-grade evidence-grounded scientific research engine with verified page citations and strict abstention safeguards.',
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#090a0d] text-[#f8fafc] antialiased selection:bg-indigo-500/30 selection:text-indigo-200">
        {children}
      </body>
    </html>
  );
}
