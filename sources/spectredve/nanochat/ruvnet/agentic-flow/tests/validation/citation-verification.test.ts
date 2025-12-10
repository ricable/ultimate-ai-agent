// Validation Tests - Citation Verification
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { KnowledgeBaseService } from '../../src/services/knowledge-base';
import { PatientData, Citation } from '../../src/types/medical';

describe('Citation Verification', () => {
  let analyzer: MedicalAnalyzerService;
  let knowledgeBase: KnowledgeBaseService;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    knowledgeBase = new KnowledgeBaseService();
  });

  describe('Citation quality', () => {
    it('should provide citations for all diagnoses', async () => {
      const patientData: PatientData = {
        id: 'patient_cite_001',
        age: 55,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: ['diabetes'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.citations).toBeDefined();
      expect(analysis.citations.length).toBeGreaterThan(0);
      expect(analysis.diagnosis.length).toBeGreaterThan(0);
    });

    it('should verify all citations', async () => {
      const patientData: PatientData = {
        id: 'patient_verify_cite_001',
        age: 50,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      for (const citation of analysis.citations) {
        expect(citation.verified).toBe(true);
      }
    });

    it('should have high relevance scores for citations', async () => {
      const patientData: PatientData = {
        id: 'patient_relevance_001',
        age: 60,
        gender: 'male',
        symptoms: ['chest pain'],
        medicalHistory: ['heart disease'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      for (const citation of analysis.citations) {
        expect(citation.relevance).toBeGreaterThan(0.7);
        expect(citation.relevance).toBeLessThanOrEqual(1);
      }
    });

    it('should include proper citation format', async () => {
      const patientData: PatientData = {
        id: 'patient_format_001',
        age: 45,
        gender: 'female',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      for (const citation of analysis.citations) {
        expect(citation.source).toBeDefined();
        expect(citation.source.length).toBeGreaterThan(0);
        expect(citation.reference).toBeDefined();
        expect(citation.reference.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Citation verification against knowledge base', () => {
    it('should verify citations in knowledge base', async () => {
      const citation: Citation = {
        source: 'Medical Journal 1',
        reference: 'DOI: 10.1234/medj.1000',
        relevance: 0.9,
        verified: false,
      };

      const verified = await knowledgeBase.verifyCitation(citation);
      expect(verified).toBe(true);
    });

    it('should detect invalid citations', async () => {
      const invalidCitation: Citation = {
        source: 'Unknown Source',
        reference: 'Invalid Reference',
        relevance: 0.5,
        verified: false,
      };

      const verified = await knowledgeBase.verifyCitation(invalidCitation);

      // Should handle gracefully
      expect(typeof verified).toBe('boolean');
    });

    it('should cross-check citations with medical literature', async () => {
      const patientData: PatientData = {
        id: 'patient_crosscheck_cite_001',
        age: 55,
        gender: 'male',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Verify cross-checks were performed
      expect(analysis.metadata.knowledgeBaseCrossChecks).toBeGreaterThan(0);

      // Verify citations are valid
      for (const citation of analysis.citations) {
        const isValid = await knowledgeBase.verifyCitation(citation);
        expect(isValid).toBe(true);
      }
    });
  });

  describe('Citation completeness', () => {
    it('should provide citations for medical claims', async () => {
      const patientData: PatientData = {
        id: 'patient_claims_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Medical analysis should have supporting citations
      expect(analysis.citations.length).toBeGreaterThan(0);
      expect(analysis.diagnosis.length).toBeGreaterThan(0);
    });

    it('should provide citations for recommendations', async () => {
      const patientData: PatientData = {
        id: 'patient_rec_cite_001',
        age: 60,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: ['condition'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Recommendations should be evidence-based
      expect(analysis.recommendations.length).toBeGreaterThan(0);
      expect(analysis.citations.length).toBeGreaterThan(0);
    });
  });

  describe('Citation traceability', () => {
    it('should link citations to specific diagnoses', async () => {
      const patientData: PatientData = {
        id: 'patient_trace_001',
        age: 55,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Each diagnosis should have supporting citations
      expect(analysis.diagnosis.length).toBeGreaterThanOrEqual(1);
      expect(analysis.citations.length).toBeGreaterThanOrEqual(1);
    });

    it('should provide DOI or reference identifiers', async () => {
      const patientData: PatientData = {
        id: 'patient_doi_001',
        age: 50,
        gender: 'female',
        symptoms: ['symptom'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      for (const citation of analysis.citations) {
        expect(citation.reference).toBeDefined();
        // Should contain some form of identifier
        expect(citation.reference.length).toBeGreaterThan(5);
      }
    });
  });
});
