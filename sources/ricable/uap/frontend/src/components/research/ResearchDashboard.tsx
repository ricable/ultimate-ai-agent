import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface ResearchPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  publication_date: string;
  venue: string;
  quality_score: number;
  novelty_score: number;
  keywords: string[];
}

interface ResearchHypothesis {
  id: string;
  text: string;
  domain: string;
  confidence: number;
  testability_score: number;
  novelty_score: number;
  validated: boolean;
}

interface LiteratureReview {
  id: string;
  topic: string;
  summary: string;
  key_findings: string[];
  research_gaps: string[];
  review_quality: number;
  generated_at: string;
}

interface ResearchInsights {
  total_papers: number;
  average_quality: number;
  average_novelty: number;
  top_venues: [string, number][];
  research_trends: string[];
  emerging_topics: string[];
}

export function ResearchDashboard() {
  const [papers, setPapers] = useState<ResearchPaper[]>([]);
  const [hypotheses, setHypotheses] = useState<ResearchHypothesis[]>([]);
  const [reviews, setReviews] = useState<LiteratureReview[]>([]);
  const [insights, setInsights] = useState<ResearchInsights | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadResearchData();
  }, []);

  const loadResearchData = async () => {
    setLoading(true);
    try {
      // In a real implementation, these would be actual API calls
      // For now, using mock data
      setPapers(mockPapers);
      setHypotheses(mockHypotheses);
      setReviews(mockReviews);
      setInsights(mockInsights);
    } catch (error) {
      console.error('Error loading research data:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateHypotheses = async (domain: string) => {
    setLoading(true);
    try {
      // API call to generate hypotheses
      const response = await fetch('/api/research/hypotheses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain, paper_ids: papers.map(p => p.id) })
      });
      
      if (response.ok) {
        const newHypotheses = await response.json();
        setHypotheses(prev => [...prev, ...newHypotheses]);
      }
    } catch (error) {
      console.error('Error generating hypotheses:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateLiteratureReview = async (topic: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/research/literature-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic })
      });
      
      if (response.ok) {
        const newReview = await response.json();
        setReviews(prev => [...prev, newReview]);
      }
    } catch (error) {
      console.error('Error generating literature review:', error);
    } finally {
      setLoading(false);
    }
  };

  const TabButton = ({ id, label, active, onClick }: {
    id: string;
    label: string;
    active: boolean;
    onClick: (id: string) => void;
  }) => (
    <button
      onClick={() => onClick(id)}
      className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${
        active 
          ? 'bg-blue-500 text-white border-b-2 border-blue-500' 
          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Advanced AI Research Platform
        </h1>
        <p className="text-gray-600">
          Automated research analysis, hypothesis generation, and literature review
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="mb-6 border-b">
        <div className="flex space-x-1">
          <TabButton 
            id="overview" 
            label="Overview" 
            active={activeTab === 'overview'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="papers" 
            label="Papers" 
            active={activeTab === 'papers'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="hypotheses" 
            label="Hypotheses" 
            active={activeTab === 'hypotheses'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="reviews" 
            label="Literature Reviews" 
            active={activeTab === 'reviews'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="insights" 
            label="Research Insights" 
            active={activeTab === 'insights'} 
            onClick={setActiveTab} 
          />
        </div>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      )}

      {/* Overview Tab */}
      {activeTab === 'overview' && !loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Total Papers</h3>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-blue-600">{papers.length}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Generated Hypotheses</h3>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-600">{hypotheses.length}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Literature Reviews</h3>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-purple-600">{reviews.length}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <h3 className="text-lg font-semibold">Avg Quality Score</h3>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-orange-600">
                {insights ? insights.average_quality.toFixed(2) : '0.00'}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Papers Tab */}
      {activeTab === 'papers' && !loading && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Research Papers</h2>
            <button className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
              Add Paper
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {papers.map(paper => (
              <PaperCard key={paper.id} paper={paper} />
            ))}
          </div>
        </div>
      )}

      {/* Hypotheses Tab */}
      {activeTab === 'hypotheses' && !loading && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Research Hypotheses</h2>
            <div className="space-x-2">
              <button 
                onClick={() => generateHypotheses('AI/ML')}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
              >
                Generate AI/ML Hypotheses
              </button>
              <button 
                onClick={() => generateHypotheses('Computer Vision')}
                className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
              >
                Generate CV Hypotheses
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {hypotheses.map(hypothesis => (
              <HypothesisCard key={hypothesis.id} hypothesis={hypothesis} />
            ))}
          </div>
        </div>
      )}

      {/* Literature Reviews Tab */}
      {activeTab === 'reviews' && !loading && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-2xl font-bold">Literature Reviews</h2>
            <div className="space-x-2">
              <button 
                onClick={() => generateLiteratureReview('Machine Learning')}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
              >
                Generate ML Review
              </button>
              <button 
                onClick={() => generateLiteratureReview('Neural Networks')}
                className="bg-indigo-500 text-white px-4 py-2 rounded hover:bg-indigo-600"
              >
                Generate NN Review
              </button>
            </div>
          </div>

          <div className="space-y-6">
            {reviews.map(review => (
              <ReviewCard key={review.id} review={review} />
            ))}
          </div>
        </div>
      )}

      {/* Research Insights Tab */}
      {activeTab === 'insights' && !loading && insights && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">Research Insights</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Top Venues</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {insights.top_venues.slice(0, 5).map(([venue, count]) => (
                    <div key={venue} className="flex justify-between">
                      <span className="text-sm text-gray-600 truncate">{venue}</span>
                      <span className="text-sm font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Research Trends</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {insights.research_trends.slice(0, 5).map((trend, index) => (
                    <div key={index} className="text-sm text-gray-600">
                      {trend}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Emerging Topics</h3>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {insights.emerging_topics.map((topic, index) => (
                    <span 
                      key={index}
                      className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">Quality Metrics</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Average Quality</span>
                      <span>{insights.average_quality.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${insights.average_quality * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Average Novelty</span>
                      <span>{insights.average_novelty.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${insights.average_novelty * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

// Paper Card Component
function PaperCard({ paper }: { paper: ResearchPaper }) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <h3 className="text-lg font-semibold line-clamp-2">{paper.title}</h3>
        <p className="text-sm text-gray-600">
          {paper.authors.slice(0, 3).join(', ')}
          {paper.authors.length > 3 && ` et al.`}
        </p>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-700 mb-3 line-clamp-3">{paper.abstract}</p>
        
        <div className="flex justify-between items-center mb-3">
          <span className="text-xs text-gray-500">{paper.venue}</span>
          <span className="text-xs text-gray-500">
            {new Date(paper.publication_date).getFullYear()}
          </span>
        </div>

        <div className="flex justify-between items-center mb-3">
          <div className="flex space-x-4">
            <div className="text-xs">
              <span className="text-gray-500">Quality:</span>
              <span className="ml-1 font-medium text-blue-600">
                {paper.quality_score.toFixed(2)}
              </span>
            </div>
            <div className="text-xs">
              <span className="text-gray-500">Novelty:</span>
              <span className="ml-1 font-medium text-green-600">
                {paper.novelty_score.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-1">
          {paper.keywords.slice(0, 3).map((keyword, index) => (
            <span 
              key={index}
              className="bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded"
            >
              {keyword}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Hypothesis Card Component
function HypothesisCard({ hypothesis }: { hypothesis: ResearchHypothesis }) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex justify-between items-start">
          <h3 className="text-lg font-semibold">Hypothesis #{hypothesis.id.slice(-6)}</h3>
          <span className={`text-xs px-2 py-1 rounded-full ${
            hypothesis.validated 
              ? 'bg-green-100 text-green-800' 
              : 'bg-yellow-100 text-yellow-800'
          }`}>
            {hypothesis.validated ? 'Validated' : 'Pending'}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-700 mb-4">{hypothesis.text}</p>
        
        <div className="grid grid-cols-2 gap-4 mb-3">
          <div className="text-xs">
            <span className="text-gray-500">Domain:</span>
            <span className="ml-1 font-medium">{hypothesis.domain}</span>
          </div>
          <div className="text-xs">
            <span className="text-gray-500">Confidence:</span>
            <span className="ml-1 font-medium text-blue-600">
              {(hypothesis.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-xs">
            <span className="text-gray-500">Testability:</span>
            <span className="ml-1 font-medium text-green-600">
              {(hypothesis.testability_score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-xs">
            <span className="text-gray-500">Novelty:</span>
            <span className="ml-1 font-medium text-purple-600">
              {(hypothesis.novelty_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="flex space-x-2">
          <button className="bg-blue-500 text-white text-xs px-3 py-1 rounded hover:bg-blue-600">
            Design Experiment
          </button>
          <button className="bg-gray-500 text-white text-xs px-3 py-1 rounded hover:bg-gray-600">
            Validate
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

// Review Card Component
function ReviewCard({ review }: { review: LiteratureReview }) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex justify-between items-start">
          <h3 className="text-lg font-semibold">{review.topic}</h3>
          <div className="text-xs text-gray-500">
            Quality: {(review.review_quality * 100).toFixed(0)}%
          </div>
        </div>
        <p className="text-xs text-gray-500">
          Generated: {new Date(review.generated_at).toLocaleDateString()}
        </p>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-700 mb-4">{review.summary}</p>
        
        <div className="mb-4">
          <h4 className="text-sm font-semibold mb-2">Key Findings</h4>
          <ul className="text-xs text-gray-600 space-y-1">
            {review.key_findings.slice(0, 3).map((finding, index) => (
              <li key={index} className="truncate">• {finding}</li>
            ))}
          </ul>
        </div>

        <div className="mb-4">
          <h4 className="text-sm font-semibold mb-2">Research Gaps</h4>
          <ul className="text-xs text-gray-600 space-y-1">
            {review.research_gaps.slice(0, 2).map((gap, index) => (
              <li key={index} className="truncate">• {gap}</li>
            ))}
          </ul>
        </div>

        <button className="bg-purple-500 text-white text-xs px-3 py-1 rounded hover:bg-purple-600">
          View Full Review
        </button>
      </CardContent>
    </Card>
  );
}

// Mock data
const mockPapers: ResearchPaper[] = [
  {
    id: '1',
    title: 'Attention Is All You Need',
    authors: ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
    abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...',
    publication_date: '2017-06-12',
    venue: 'NIPS 2017',
    quality_score: 0.95,
    novelty_score: 0.98,
    keywords: ['transformers', 'attention', 'neural networks']
  },
  {
    id: '2',
    title: 'BERT: Pre-training of Deep Bidirectional Transformers',
    authors: ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
    abstract: 'We introduce a new language representation model called BERT...',
    publication_date: '2018-10-11',
    venue: 'NAACL 2019',
    quality_score: 0.92,
    novelty_score: 0.89,
    keywords: ['BERT', 'pretraining', 'bidirectional']
  }
];

const mockHypotheses: ResearchHypothesis[] = [
  {
    id: '1',
    text: 'If attention mechanisms are applied to graph neural networks, then node classification accuracy will increase significantly.',
    domain: 'AI/ML',
    confidence: 0.8,
    testability_score: 0.9,
    novelty_score: 0.7,
    validated: false
  },
  {
    id: '2',
    text: 'Multimodal transformers will outperform unimodal models in visual question answering tasks.',
    domain: 'Computer Vision',
    confidence: 0.75,
    testability_score: 0.85,
    novelty_score: 0.65,
    validated: true
  }
];

const mockReviews: LiteratureReview[] = [
  {
    id: '1',
    topic: 'Transformer Architectures in NLP',
    summary: 'This review examines the evolution and impact of transformer architectures in natural language processing...',
    key_findings: [
      'Transformers revolutionized sequence modeling',
      'Self-attention mechanisms are highly effective',
      'Pre-training strategies are crucial for performance'
    ],
    research_gaps: [
      'Limited work on efficiency improvements',
      'Few studies on interpretability'
    ],
    review_quality: 0.88,
    generated_at: '2024-01-15T10:30:00Z'
  }
];

const mockInsights: ResearchInsights = {
  total_papers: 1247,
  average_quality: 0.78,
  average_novelty: 0.72,
  top_venues: [
    ['NIPS', 156],
    ['ICML', 134],
    ['ICLR', 121],
    ['ACL', 89],
    ['EMNLP', 76]
  ],
  research_trends: [
    'Transformer models: 5 years of publications',
    'Self-supervised learning: 4 years of growth',
    'Multimodal AI: emerging trend'
  ],
  emerging_topics: [
    'foundation models',
    'prompt engineering',
    'in-context learning',
    'retrieval augmentation'
  ]
};