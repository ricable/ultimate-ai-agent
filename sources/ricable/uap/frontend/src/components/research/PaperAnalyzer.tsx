import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface PaperAnalysis {
  paper_id: string;
  quality_assessment: {
    overall_quality: number;
    abstract_quality: number;
    methodology_clarity: number;
    citation_quality: number;
    writing_quality: number;
  };
  novelty_assessment: number;
  domain_classification: { [key: string]: number };
  methodology_analysis: {
    experimental_design: string;
    sample_size: number | null;
    statistical_tests: string[];
    datasets_used: string[];
    evaluation_metrics: string[];
  };
  reproducibility_score: number;
  impact_prediction: {
    citation_potential: number;
    practical_applicability: number;
    theoretical_significance: number;
    industry_relevance: number;
  };
  extracted_concepts: string[];
  research_gaps: string[];
}

interface PaperAnalyzerProps {
  onAnalysisComplete?: (analysis: PaperAnalysis) => void;
}

export function PaperAnalyzer({ onAnalysisComplete }: PaperAnalyzerProps) {
  const [file, setFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<PaperAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const analyzePaper = async () => {
    if (!file) {
      setError('Please select a paper to analyze');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('paper', file);

      const response = await fetch('/api/research/analyze-paper', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to analyze paper');
      }

      const analysis = await response.json();
      setAnalysisResult(analysis);
      
      if (onAnalysisComplete) {
        onAnalysisComplete(analysis);
      }
    } catch (err) {
      setError('Error analyzing paper. Please try again.');
      console.error('Paper analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const QualityBar = ({ label, value, color = 'blue' }: {
    label: string;
    value: number;
    color?: string;
  }) => (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="font-medium">{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`bg-${color}-600 h-2 rounded-full transition-all duration-300`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );

  const ScoreCard = ({ title, score, description }: {
    title: string;
    score: number;
    description: string;
  }) => (
    <Card className="h-full">
      <CardHeader>
        <h3 className="text-lg font-semibold">{title}</h3>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold text-blue-600 mb-2">
          {(score * 100).toFixed(0)}%
        </div>
        <p className="text-sm text-gray-600">{description}</p>
      </CardContent>
    </Card>
  );

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Research Paper Analyzer
        </h1>
        <p className="text-gray-600">
          Upload a research paper for comprehensive quality and novelty analysis
        </p>
      </div>

      {/* File Upload Section */}
      <Card className="mb-8">
        <CardHeader>
          <h2 className="text-xl font-semibold">Upload Paper</h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <input
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={handleFileUpload}
                className="hidden"
                id="paper-upload"
              />
              <label 
                htmlFor="paper-upload" 
                className="cursor-pointer flex flex-col items-center space-y-2"
              >
                <div className="text-4xl text-gray-400">📄</div>
                <div className="text-lg font-medium text-gray-700">
                  Click to upload paper
                </div>
                <div className="text-sm text-gray-500">
                  Supports PDF, DOCX, and TXT files
                </div>
              </label>
            </div>

            {file && (
              <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                <span className="text-sm text-gray-700">{file.name}</span>
                <button
                  onClick={() => setFile(null)}
                  className="text-red-500 hover:text-red-700"
                >
                  Remove
                </button>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}

            <button
              onClick={analyzePaper}
              disabled={!file || loading}
              className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Analyze Paper'}
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-lg text-gray-600">Analyzing paper...</span>
        </div>
      )}

      {/* Analysis Results */}
      {analysisResult && !loading && (
        <div className="space-y-8">
          {/* Overview Scores */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <ScoreCard
              title="Overall Quality"
              score={analysisResult.quality_assessment.overall_quality}
              description="Comprehensive quality assessment based on multiple factors"
            />
            <ScoreCard
              title="Novelty Score"
              score={analysisResult.novelty_assessment}
              description="How novel and innovative the research contribution is"
            />
            <ScoreCard
              title="Reproducibility"
              score={analysisResult.reproducibility_score}
              description="How easily the research can be reproduced"
            />
          </div>

          {/* Quality Assessment Details */}
          <Card>
            <CardHeader>
              <h2 className="text-xl font-semibold">Quality Assessment Breakdown</h2>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <QualityBar
                    label="Abstract Quality"
                    value={analysisResult.quality_assessment.abstract_quality}
                    color="green"
                  />
                  <QualityBar
                    label="Methodology Clarity"
                    value={analysisResult.quality_assessment.methodology_clarity}
                    color="blue"
                  />
                  <QualityBar
                    label="Citation Quality"
                    value={analysisResult.quality_assessment.citation_quality}
                    color="purple"
                  />
                </div>
                <div>
                  <QualityBar
                    label="Writing Quality"
                    value={analysisResult.quality_assessment.writing_quality}
                    color="orange"
                  />
                  <QualityBar
                    label="Overall Quality"
                    value={analysisResult.quality_assessment.overall_quality}
                    color="red"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Domain Classification */}
          <Card>
            <CardHeader>
              <h2 className="text-xl font-semibold">Domain Classification</h2>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(analysisResult.domain_classification).map(([domain, confidence]) => (
                  <div key={domain} className="text-center">
                    <div className="text-lg font-semibold text-blue-600">
                      {(confidence * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-600 capitalize">
                      {domain.replace('_', ' ')}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Impact Prediction */}
          <Card>
            <CardHeader>
              <h2 className="text-xl font-semibold">Impact Prediction</h2>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <QualityBar
                  label="Citation Potential"
                  value={analysisResult.impact_prediction.citation_potential}
                  color="green"
                />
                <QualityBar
                  label="Practical Applicability"
                  value={analysisResult.impact_prediction.practical_applicability}
                  color="blue"
                />
                <QualityBar
                  label="Theoretical Significance"
                  value={analysisResult.impact_prediction.theoretical_significance}
                  color="purple"
                />
                <QualityBar
                  label="Industry Relevance"
                  value={analysisResult.impact_prediction.industry_relevance}
                  color="orange"
                />
              </div>
            </CardContent>
          </Card>

          {/* Methodology Analysis */}
          <Card>
            <CardHeader>
              <h2 className="text-xl font-semibold">Methodology Analysis</h2>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-2">Experimental Design</h3>
                  <p className="text-gray-700 mb-4">
                    {analysisResult.methodology_analysis.experimental_design}
                  </p>
                  
                  {analysisResult.methodology_analysis.sample_size && (
                    <div className="mb-4">
                      <h4 className="font-medium">Sample Size</h4>
                      <p className="text-gray-700">
                        {analysisResult.methodology_analysis.sample_size}
                      </p>
                    </div>
                  )}
                </div>
                
                <div>
                  <h3 className="font-semibold mb-2">Statistical Tests</h3>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {analysisResult.methodology_analysis.statistical_tests.map((test, index) => (
                      <span 
                        key={index}
                        className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
                      >
                        {test}
                      </span>
                    ))}
                  </div>
                  
                  <h3 className="font-semibold mb-2">Evaluation Metrics</h3>
                  <div className="flex flex-wrap gap-2">
                    {analysisResult.methodology_analysis.evaluation_metrics.map((metric, index) => (
                      <span 
                        key={index}
                        className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded"
                      >
                        {metric}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Key Concepts */}
          <Card>
            <CardHeader>
              <h2 className="text-xl font-semibold">Extracted Concepts</h2>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysisResult.extracted_concepts.map((concept, index) => (
                  <span 
                    key={index}
                    className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm"
                  >
                    {concept}
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Research Gaps */}
          {analysisResult.research_gaps.length > 0 && (
            <Card>
              <CardHeader>
                <h2 className="text-xl font-semibold">Identified Research Gaps</h2>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {analysisResult.research_gaps.map((gap, index) => (
                    <li key={index} className="text-gray-700 border-l-4 border-blue-500 pl-4">
                      {gap}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-4">
            <button className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600">
              Generate Hypotheses
            </button>
            <button className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600">
              Create Literature Review
            </button>
            <button className="bg-purple-500 text-white px-6 py-2 rounded hover:bg-purple-600">
              Conduct Peer Review
            </button>
            <button className="bg-gray-500 text-white px-6 py-2 rounded hover:bg-gray-600">
              Export Analysis
            </button>
          </div>
        </div>
      )}
    </div>
  );
}