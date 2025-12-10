/**
 * Analyzer Tests
 * Test suite for symptom analyzer
 *
 * @author ruv (github.com/ruvnet, ruv.io)
 */

import { describe, it, expect } from '@jest/globals';
import Analyzer from '../src/analyzer.js';

describe('Analyzer', () => {
  let analyzer;

  beforeEach(() => {
    analyzer = new Analyzer({
      minConfidenceScore: 0.95,
      verificationLevel: 'moderate'
    });
  });

  describe('analyze()', () => {
    it('should analyze symptoms correctly', async () => {
      const result = await analyzer.analyze({
        symptoms: 'fever, cough',
        duration: '3 days'
      });

      expect(result).toBeDefined();
      expect(result.symptoms).toBe('fever, cough');
      expect(result.disclaimer).toContain('supplement');
    });

    it('should require symptoms parameter', async () => {
      await expect(analyzer.analyze({})).rejects.toThrow();
    });
  });

  describe('assessUrgency()', () => {
    it('should return valid urgency level', () => {
      const analysis = { symptoms: 'fever' };
      const urgency = analyzer.assessUrgency(analysis);

      expect(['emergency', 'urgent', 'routine', 'self-care']).toContain(urgency);
    });
  });

  describe('verify()', () => {
    it('should verify analysis results', async () => {
      const analysis = {
        symptoms: 'fever',
        confidence: 0.9
      };

      const verification = await analyzer.verify(analysis);

      expect(verification).toBeDefined();
      expect(verification).toHaveProperty('verified');
      expect(verification).toHaveProperty('confidence');
    });
  });
});
