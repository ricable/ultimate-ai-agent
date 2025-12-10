import { useState, useRef } from 'react'
interface TiltOptions {
  max: number
  scale: number
  speed: number
  perspective: number
  easing: string
}
export const useCardTilt = (options: Partial<TiltOptions> = {}) => {
  const {
    max = 15,
    scale = 1.05,
    speed = 500,
    perspective = 1000,
    easing = 'cubic-bezier(.03,.98,.52,.99)',
  } = options
  const [tiltStyles, setTiltStyles] = useState({
    transform: `perspective(${perspective}px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`,
    transition: `transform ${speed}ms ${easing}`,
    reflectionOpacity: 0,
    reflectionPosition: '50% 50%',
    glowIntensity: 0,
    glowPosition: { x: 50, y: 50 },
  })
  const cardRef = useRef<HTMLDivElement>(null)
  const isHovering = useRef(false)
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const centerX = rect.width / 2
    const centerY = rect.height / 2
    const percentX = (x - centerX) / centerX
    const percentY = (y - centerY) / centerY
    const tiltX = max * -1 * percentY
    const tiltY = max * percentX
    // Calculate glow position (0-100%)
    const glowX = (x / rect.width) * 100
    const glowY = (y / rect.height) * 100
    // Calculate reflection position
    const reflectionX = 50 + percentX * 15
    const reflectionY = 50 + percentY * 15
    setTiltStyles({
      transform: `perspective(${perspective}px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) scale3d(${scale}, ${scale}, ${scale})`,
      transition: `transform ${speed}ms ${easing}`,
      reflectionOpacity: 0.15,
      reflectionPosition: `${reflectionX}% ${reflectionY}%`,
      glowIntensity: 1,
      glowPosition: { x: glowX, y: glowY },
    })
  }
  const handleMouseEnter = () => {
    isHovering.current = true
  }
  const handleMouseLeave = () => {
    isHovering.current = false
    setTiltStyles({
      transform: `perspective(${perspective}px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`,
      transition: `transform ${speed}ms ${easing}`,
      reflectionOpacity: 0,
      reflectionPosition: '50% 50%',
      glowIntensity: 0,
      glowPosition: { x: 50, y: 50 },
    })
  }
  const handleClick = () => {
    // Bounce animation on click
    if (cardRef.current) {
      cardRef.current.style.animation = 'card-bounce 0.4s'
      cardRef.current.addEventListener(
        'animationend',
        () => {
          if (cardRef.current) {
            cardRef.current.style.animation = ''
          }
        },
        { once: true },
      )
    }
  }
  return {
    cardRef,
    tiltStyles,
    handlers: {
      onMouseMove: handleMouseMove,
      onMouseEnter: handleMouseEnter,
      onMouseLeave: handleMouseLeave,
      onClick: handleClick,
    },
  }
}
