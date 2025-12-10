import { useEffect, useState } from 'react'
import { X, BarChart, AlertCircle, CheckCircle, XCircle, Activity, RefreshCw, ExternalLink, TestTube, Target, ChevronDown } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { CoverageVisualization, CoverageData } from './CoverageVisualization'

interface TestResults {
  summary: {
    total: number
    passed: number
    failed: number
    skipped: number
    duration: number
  }
  suites: Array<{
    name: string
    tests: number
    passed: number
    failed: number
    skipped: number
    duration: number
    failedTests?: Array<{
      name: string
      error?: string
    }>
  }>
}

// Using CoverageData from CoverageVisualization component instead
type CoverageSummary = CoverageData

interface TestResultsModalProps {
  isOpen: boolean
  onClose: () => void
}

export function TestResultsModal({ isOpen, onClose }: TestResultsModalProps) {
  const [testResults, setTestResults] = useState<TestResults | null>(null)
  const [coverage, setCoverage] = useState<CoverageSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedSuites, setExpandedSuites] = useState<Set<number>>(new Set())

  const fetchResults = async () => {
    setLoading(true)
    setError(null)
    
    console.log('[TEST RESULTS MODAL] Fetching test results...')
    
    // Add retry logic for file reading
    const fetchWithRetry = async (url: string, retries = 3): Promise<Response | null> => {
      for (let i = 0; i < retries; i++) {
        try {
          const response = await fetch(url)
          if (response.ok) {
            const text = await response.text()
            if (text && text.trim().length > 0) {
              return response
            }
          }
          // Wait a bit before retrying
          if (i < retries - 1) {
            await new Promise(resolve => setTimeout(resolve, 500))
          }
        } catch (err) {
          console.log(`[TEST RESULTS MODAL] Attempt ${i + 1} failed for ${url}:`, err)
        }
      }
      return null
    }
    
    try {
      // Fetch test results JSON with retry
      const testResponse = await fetchWithRetry('/test-results/test-results.json')
      console.log('[TEST RESULTS MODAL] Test results response:', testResponse?.status, testResponse?.statusText)
      
      if (testResponse && testResponse.ok) {
        try {
          const testData = await testResponse.json()
          console.log('[TEST RESULTS MODAL] Test data loaded:', testData)
        
          // Parse vitest results format - handle both full format and simplified format
          const results: TestResults = {
            summary: {
              total: testData.numTotalTests || 0,
              passed: testData.numPassedTests || 0,
              failed: testData.numFailedTests || 0,
              skipped: testData.numSkippedTests || testData.numPendingTests || 0,
              duration: testData.testResults?.reduce((acc: number, suite: any) => {
                const duration = suite.perfStats ? 
                  (suite.perfStats.end - suite.perfStats.start) : 
                  (suite.endTime - suite.startTime) || 0
                return acc + duration
              }, 0) || 0
            },
            suites: testData.testResults?.map((suite: any) => {
              const suiteName = suite.name?.replace(process.cwd(), '') || 
                               suite.displayName || 
                               suite.testFilePath || 
                               'Unknown'
              
              return {
                name: suiteName,
                tests: suite.numTotalTests || suite.assertionResults?.length || 0,
                passed: suite.numPassedTests || suite.assertionResults?.filter((t: any) => t.status === 'passed').length || 0,
                failed: suite.numFailedTests || suite.assertionResults?.filter((t: any) => t.status === 'failed').length || 0,
                skipped: suite.numSkippedTests || suite.numPendingTests || suite.assertionResults?.filter((t: any) => t.status === 'skipped' || t.status === 'pending').length || 0,
                duration: suite.perfStats ? 
                  (suite.perfStats.end - suite.perfStats.start) : 
                  (suite.endTime - suite.startTime) || 0,
                failedTests: (suite.assertionResults || suite.testResults)?.filter((test: any) => test.status === 'failed')
                  .map((test: any) => ({
                    name: test.title || test.fullTitle || test.ancestorTitles?.join(' > ') || 'Unknown test',
                    error: test.failureMessages?.[0] || test.error?.message || test.message || 'No error message'
                  })) || []
              }
            }) || []
          }
          setTestResults(results)
        } catch (parseError) {
          console.error('[TEST RESULTS MODAL] JSON parse error:', parseError)
          // Don't throw, just log and continue to coverage
        }
      }

      // Fetch coverage data with retry
      const coverageResponse = await fetchWithRetry('/test-results/coverage/coverage-summary.json')
      console.log('[TEST RESULTS MODAL] Coverage response:', coverageResponse?.status, coverageResponse?.statusText)
      
      if (coverageResponse && coverageResponse.ok) {
        try {
          const coverageData = await coverageResponse.json()
          console.log('[TEST RESULTS MODAL] Coverage data loaded:', coverageData)
          setCoverage(coverageData)
        } catch (parseError) {
          console.error('[TEST RESULTS MODAL] Coverage parse error:', parseError)
        }
      }

      if (!testResponse && !coverageResponse) {
        console.log('[TEST RESULTS MODAL] No data available - both requests failed')
        throw new Error('No test results or coverage data available. Please run tests first.')
      }

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load test results'
      setError(message)
      console.error('Test results fetch error:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      // Add a longer delay to ensure files are fully written
      const timer = setTimeout(() => {
        fetchResults()
      }, 1000)
      
      return () => clearTimeout(timer)
    }
  }, [isOpen])

  const getHealthScore = () => {
    if (!testResults || !coverage) return 0
    
    const testScore = testResults.summary.total > 0 
      ? (testResults.summary.passed / testResults.summary.total) * 100 
      : 0
    const coverageScore = coverage.total.lines.pct
    
    return Math.round((testScore + coverageScore) / 2)
  }

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-500'
    if (score >= 60) return 'text-yellow-500'
    return 'text-red-500'
  }

  const getHealthBg = (score: number) => {
    if (score >= 80) return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
    if (score >= 60) return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
    return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
  }

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  if (!isOpen) return null

  const healthScore = getHealthScore()

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 w-full max-w-4xl max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <TestTube className="w-6 h-6 text-blue-500" />
              <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
                Test Results Report
              </h2>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
            {loading && (
              <div className="flex items-center justify-center py-12">
                <div className="flex items-center gap-3">
                  <Activity className="w-5 h-5 animate-pulse text-blue-500" />
                  <span className="text-gray-600 dark:text-gray-400">Loading test results...</span>
                </div>
              </div>
            )}

            {error && !testResults && !coverage && (
              <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <AlertCircle className="w-12 h-12 text-yellow-500" />
                <div className="text-center">
                  <p className="text-gray-600 dark:text-gray-400 mb-2">No test results available</p>
                  <p className="text-sm text-gray-500 dark:text-gray-500">Run tests to generate the report</p>
                </div>
              </div>
            )}

            {(testResults || coverage) && (
              <div className="space-y-6">
                {/* Health Score */}
                <div className={`p-6 rounded-lg border ${getHealthBg(healthScore)}`}>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-3">
                      <Target className={`w-8 h-8 ${getHealthColor(healthScore)}`} />
                      <div>
                        <h3 className="text-xl font-bold text-gray-800 dark:text-white">
                          Test Health Score
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Overall test and coverage quality
                        </p>
                      </div>
                    </div>
                    <div className="ml-auto text-right">
                      <div className={`text-4xl font-bold ${getHealthColor(healthScore)}`}>
                        {healthScore}%
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {healthScore >= 80 ? 'Excellent' : healthScore >= 60 ? 'Good' : 'Needs Work'}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Test Results Summary */}
                  {testResults && (
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center gap-3 mb-4">
                        <TestTube className="w-5 h-5 text-blue-500" />
                        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
                          Test Summary
                        </h3>
                      </div>

                      {/* Overall Stats */}
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg border">
                          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                            {testResults.summary.passed}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">Passed</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg border">
                          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                            {testResults.summary.failed}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">Failed</div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg border">
                          <div className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                            {testResults.summary.total}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">Total Tests</div>
                        </div>
                        <div className="text-center p-3 bg-white dark:bg-gray-800 rounded-lg border">
                          <div className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                            {formatDuration(testResults.summary.duration)}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">Duration</div>
                        </div>
                      </div>

                      {/* Test Suites */}
                      <div className="space-y-2 max-h-60 overflow-y-auto">
                        {testResults.suites.map((suite, index) => (
                          <div key={index} className="bg-white dark:bg-gray-800 rounded border">
                            <div 
                              className="flex items-center justify-between p-2 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50"
                              onClick={() => {
                                if (suite.failed > 0) {
                                  const newExpanded = new Set(expandedSuites)
                                  if (newExpanded.has(index)) {
                                    newExpanded.delete(index)
                                  } else {
                                    newExpanded.add(index)
                                  }
                                  setExpandedSuites(newExpanded)
                                }
                              }}
                            >
                              <div className="flex items-center gap-2">
                                {suite.failed > 0 ? (
                                  <XCircle className="w-4 h-4 text-red-500" />
                                ) : (
                                  <CheckCircle className="w-4 h-4 text-green-500" />
                                )}
                                <span className="font-mono text-xs truncate max-w-[200px]" title={suite.name}>
                                  {suite.name.split('/').pop()}
                                </span>
                              </div>
                              <div className="flex items-center gap-2 text-xs">
                                <span className="text-green-600">{suite.passed}</span>
                                <span className="text-gray-400">/</span>
                                <span className="text-red-600">{suite.failed}</span>
                                <span className="text-gray-500">({formatDuration(suite.duration)})</span>
                                {suite.failed > 0 && (
                                  <motion.div
                                    animate={{ rotate: expandedSuites.has(index) ? 180 : 0 }}
                                    transition={{ duration: 0.2 }}
                                  >
                                    <ChevronDown className="w-3 h-3 text-gray-400" />
                                  </motion.div>
                                )}
                              </div>
                            </div>
                            
                            {/* Expandable failed tests */}
                            <AnimatePresence>
                              {expandedSuites.has(index) && suite.failedTests && suite.failedTests.length > 0 && (
                                <motion.div
                                  initial={{ height: 0, opacity: 0 }}
                                  animate={{ height: 'auto', opacity: 1 }}
                                  exit={{ height: 0, opacity: 0 }}
                                  transition={{ duration: 0.2 }}
                                  className="border-t border-gray-200 dark:border-gray-700 overflow-hidden"
                                >
                                  <div className="p-2 space-y-2 bg-red-50 dark:bg-red-900/10">
                                    {suite.failedTests.map((test, testIndex) => (
                                      <div key={testIndex} className="space-y-1">
                                        <div className="flex items-start gap-2">
                                          <XCircle className="w-3 h-3 text-red-500 mt-0.5 flex-shrink-0" />
                                          <div className="flex-1">
                                            <p className="text-xs font-medium text-red-700 dark:text-red-400">
                                              {test.name}
                                            </p>
                                            {test.error && (
                                              <pre className="mt-1 text-xs text-red-600 dark:text-red-500 whitespace-pre-wrap font-mono bg-red-100 dark:bg-red-900/20 p-2 rounded">
                                                {test.error}
                                              </pre>
                                            )}
                                          </div>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Coverage Visualization */}
                  {coverage && (
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
                      <CoverageVisualization 
                        coverage={coverage} 
                        compact={true}
                        showFileBreakdown={false}
                      />
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <button
                    onClick={fetchResults}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg transition-colors"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Refresh
                  </button>
                  <button
                    onClick={() => window.open('/api/coverage/pytest/html/index.html', '_blank')}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                  >
                    <ExternalLink className="w-4 h-4" />
                    Detailed Report
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
} 