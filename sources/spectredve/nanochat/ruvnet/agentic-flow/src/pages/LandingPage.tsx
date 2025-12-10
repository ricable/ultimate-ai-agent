import Hero from '../components/Hero';
import Features from '../components/Features';
import CoreComponents from '../components/CoreComponents';
import PerformanceMetrics from '../components/PerformanceMetrics';
import QuickStart from '../components/QuickStart';
import AgentTypes from '../components/AgentTypes';
import Footer from '../components/Footer';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-hero">
      <Hero />
      <Features />
      <CoreComponents />
      <PerformanceMetrics />
      <QuickStart />
      <AgentTypes />
      <Footer />
    </div>
  );
};

export default LandingPage;
