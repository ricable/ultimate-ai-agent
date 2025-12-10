import { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <main className="container mx-auto max-w-7xl">
        {children}
      </main>
    </div>
  )
}