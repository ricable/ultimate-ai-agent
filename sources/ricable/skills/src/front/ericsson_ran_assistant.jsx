import React, { useState, useEffect } from 'react';
import { Search, Settings, Zap, AlertCircle, ChevronDown, ChevronUp, Copy, Check, Book } from 'lucide-react';

const EriccssonRANAssistant = () => {
  const [activeTab, setActiveTab] = useState('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [copied, setCopied] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});

  // Sample data structure based on the Ericsson RAN knowledge base
  const sampleFeatures = [
    {
      id: 'FAJ_121_3055',
      name: 'Multi-Operator RAN',
      fajId: 'FAJ 121 3055',
      cxcCode: 'CXC4011512',
      accessType: 'LTE',
      nodeType: 'Baseband Radio Node',
      valuePackage: 'Dual-eNodeB Multioperator RAN',
      description: 'Enable multiple operators to share radio resources on the same Baseband Radio Node',
      activation: 'Set FeatureState.featureState to ACTIVATED in FeatureState=CXC4011512 MO instance',
      deactivation: 'Set FeatureState.featureState to DEACTIVATED in FeatureState=CXC4011512 MO instance',
      parameters: [
        { name: 'ENodeBFunction.timeAndPhaseSynchCritical', type: 'Introduced', moClass: 'ENodeBFunction' },
        { name: 'SectorCarrier.configuredMaxTxPower', type: 'Affected', moClass: 'SectorCarrier', note: 'Limited by power config of other LTE node' },
        { name: 'SectorEquipmentFunction.availableHwOutputPower', type: 'Affected', moClass: 'SectorEquipmentFunction', note: 'Total power available for different LTE nodes' }
      ],
      counters: ['pment', 'pmentFunction'],
      prerequisites: ['Dual eNodeB support', 'Shared radio hardware', 'Synchronized timing'],
      impact: 'Allows spectrum sharing, reduces capex and opex',
      bestPractices: [
        'Ensure timing synchronization between operators',
        'Configure power limits carefully for each operator',
        'Monitor interference levels regularly',
        'Coordinate maintenance windows with all operators'
      ]
    },
    {
      id: 'FAJ_121_3094',
      name: 'MIMO Sleep Mode',
      fajId: 'FAJ 121 3094',
      cxcCode: 'CXC4011808',
      accessType: 'LTE/NR',
      nodeType: 'Radio Node',
      valuePackage: 'Energy Efficiency',
      description: 'Reduce energy consumption by deactivating MIMO antennas during low traffic periods',
      activation: 'Set MimoSleepFunction.mimoSleepMode to ENABLED',
      deactivation: 'Set MimoSleepFunction.mimoSleepMode to DISABLED',
      parameters: [
        { name: 'MimoSleepFunction.mimoSleepMode', type: 'Configuration', moClass: 'MimoSleepFunction' },
        { name: 'MimoSleepFunction.sleepThreshold', type: 'Configuration', moClass: 'MimoSleepFunction', note: 'Traffic threshold before sleeping' },
        { name: 'MimoSleepFunction.wakeupTime', type: 'Configuration', moClass: 'MimoSleepFunction', note: 'Time to return to full MIMO' }
      ],
      counters: ['pmMimoSleepTime', 'pmMimoWakeups'],
      prerequisites: ['Dual antenna support', 'Traffic monitoring enabled'],
      impact: '15-25% energy savings during low traffic periods',
      bestPractices: [
        'Set sleep threshold based on your traffic patterns',
        'Monitor wake-up delays during peak hours',
        'Adjust wakeup time for optimal performance',
        'Review energy savings weekly'
      ]
    }
  ];

  const categories = [
    { id: 'carrier-agg', name: 'Carrier Aggregation', count: 25 },
    { id: 'dual-conn', name: 'Dual Connectivity', count: 3 },
    { id: 'energy', name: 'Energy Efficiency', count: 2 },
    { id: 'mimo', name: 'MIMO Features', count: 6 },
    { id: 'mobility', name: 'Mobility', count: 27 },
    { id: 'other', name: 'Other', count: 314 }
  ];

  const filteredFeatures = sampleFeatures.filter(f =>
    f.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    f.fajId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    f.cxcCode.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const SearchTab = () => (
    <div className="space-y-6">
      <div>
        <div className="relative">
          <Search className="absolute left-3 top-3 text-gray-400" size={20} />
          <input
            type="text"
            placeholder="Chercher feature, FAJ ID, CXC code..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {categories.map(cat => (
          <button
            key={cat.id}
            className="p-3 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg hover:shadow-md transition text-left"
          >
            <div className="font-semibold text-sm text-gray-800">{cat.name}</div>
            <div className="text-xs text-gray-600 mt-1">{cat.count} features</div>
          </button>
        ))}
      </div>

      <div className="space-y-3">
        <h3 className="font-bold text-gray-800">Résultats ({filteredFeatures.length})</h3>
        {filteredFeatures.map(feature => (
          <button
            key={feature.id}
            onClick={() => setSelectedFeature(feature)}
            className="w-full p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition text-left"
          >
            <div className="font-semibold text-gray-900">{feature.name}</div>
            <div className="text-sm text-gray-600 mt-1">
              {feature.fajId} • {feature.cxcCode} • {feature.accessType}
            </div>
          </button>
        ))}
      </div>
    </div>
  );

  const FeatureDetailTab = () => (
    <div className="space-y-6">
      {selectedFeature ? (
        <>
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 rounded-lg">
            <h2 className="text-2xl font-bold mb-2">{selectedFeature.name}</h2>
            <p className="text-blue-100">{selectedFeature.description}</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-xs text-gray-600 uppercase tracking-wide">FAJ ID</div>
              <div className="font-mono font-bold text-gray-900 mt-1">{selectedFeature.fajId}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-xs text-gray-600 uppercase tracking-wide">CXC Code</div>
              <div className="font-mono font-bold text-gray-900 mt-1">{selectedFeature.cxcCode}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-xs text-gray-600 uppercase tracking-wide">Access</div>
              <div className="font-bold text-gray-900 mt-1">{selectedFeature.accessType}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-xs text-gray-600 uppercase tracking-wide">Node Type</div>
              <div className="font-bold text-gray-900 mt-1 text-sm">{selectedFeature.nodeType}</div>
            </div>
          </div>

          {/* Activation Section */}
          <div className="border border-gray-200 rounded-lg">
            <button
              onClick={() => toggleSection('activation')}
              className="w-full flex justify-between items-center p-4 hover:bg-gray-50"
            >
              <div className="flex items-center gap-2">
                <Zap size={18} className="text-green-600" />
                <span className="font-semibold text-gray-900">Activation</span>
              </div>
              {expandedSections.activation ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>
            {expandedSections.activation && (
              <div className="px-4 pb-4 border-t border-gray-200 bg-green-50">
                <p className="text-sm text-gray-700 mb-3">{selectedFeature.activation}</p>
                <button
                  onClick={() => copyToClipboard(selectedFeature.activation, 'activation')}
                  className="flex items-center gap-2 text-sm text-green-700 hover:text-green-900"
                >
                  {copied === 'activation' ? <Check size={16} /> : <Copy size={16} />}
                  {copied === 'activation' ? 'Copié!' : 'Copier'}
                </button>
              </div>
            )}
          </div>

          {/* Parameters Section */}
          <div className="border border-gray-200 rounded-lg">
            <button
              onClick={() => toggleSection('parameters')}
              className="w-full flex justify-between items-center p-4 hover:bg-gray-50"
            >
              <div className="flex items-center gap-2">
                <Settings size={18} className="text-blue-600" />
                <span className="font-semibold text-gray-900">Paramètres ({selectedFeature.parameters.length})</span>
              </div>
              {expandedSections.parameters ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>
            {expandedSections.parameters && (
              <div className="px-4 pb-4 border-t border-gray-200 space-y-3 bg-blue-50">
                {selectedFeature.parameters.map((param, idx) => (
                  <div key={idx} className="bg-white p-3 rounded border border-blue-100">
                    <div className="font-mono text-sm font-semibold text-gray-900">{param.name}</div>
                    <div className="text-xs text-gray-600 mt-1">
                      <span className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">
                        {param.moClass}
                      </span>
                      <span className="inline-block bg-gray-100 text-gray-800 px-2 py-1 rounded">
                        {param.type}
                      </span>
                    </div>
                    {param.note && <p className="text-xs text-gray-600 mt-2 italic">{param.note}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Best Practices Section */}
          <div className="border border-gray-200 rounded-lg">
            <button
              onClick={() => toggleSection('practices')}
              className="w-full flex justify-between items-center p-4 hover:bg-gray-50"
            >
              <div className="flex items-center gap-2">
                <Book size={18} className="text-amber-600" />
                <span className="font-semibold text-gray-900">Best Practices</span>
              </div>
              {expandedSections.practices ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>
            {expandedSections.practices && (
              <div className="px-4 pb-4 border-t border-gray-200 space-y-2 bg-amber-50">
                {selectedFeature.bestPractices.map((practice, idx) => (
                  <div key={idx} className="flex gap-3">
                    <div className="text-amber-600 font-bold">•</div>
                    <p className="text-sm text-gray-700">{practice}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Impact Section */}
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 p-4 rounded-lg border border-amber-200">
            <div className="flex gap-3">
              <AlertCircle className="text-orange-600 flex-shrink-0" size={20} />
              <div>
                <div className="font-semibold text-gray-900 mb-1">Impact Réseau</div>
                <p className="text-sm text-gray-700">{selectedFeature.impact}</p>
              </div>
            </div>
          </div>

          <button
            onClick={() => setSelectedFeature(null)}
            className="w-full py-2 text-blue-600 hover:text-blue-800 font-semibold"
          >
            ← Retour à la recherche
          </button>
        </>
      ) : (
        <div className="text-center py-12 text-gray-500">
          <p>Sélectionnez une feature pour voir les détails</p>
        </div>
      )}
    </div>
  );

  const ConfigurationTab = () => (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
        <h3 className="font-bold text-gray-900 mb-3">Configuration Helper</h3>
        <div className="space-y-4">
          <div className="bg-white p-4 rounded border border-blue-100">
            <div className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
              <Zap size={16} className="text-green-600" />
              Étapes d'Activation Générales
            </div>
            <ol className="text-sm text-gray-700 space-y-2 ml-4">
              <li>1. Vérifier les prérequis techniques</li>
              <li>2. Analyser l'impact sur le réseau</li>
              <li>3. Planifier le déploiement</li>
              <li>4. Tester en environnement de test</li>
              <li>5. Obtenir les approbations nécessaires</li>
              <li>6. Exécuter l'activation en production</li>
              <li>7. Monitorer les performances</li>
            </ol>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
          <h4 className="font-semibold text-green-900 mb-2">✓ Before Activation</h4>
          <ul className="text-sm text-green-800 space-y-1">
            <li>• Check compatibility</li>
            <li>• Backup configuration</li>
            <li>• Notify operations team</li>
            <li>• Schedule maintenance window</li>
          </ul>
        </div>
        <div className="bg-orange-50 border border-orange-200 p-4 rounded-lg">
          <h4 className="font-semibold text-orange-900 mb-2">⚠ After Activation</h4>
          <ul className="text-sm text-orange-800 space-y-1">
            <li>• Verify feature state</li>
            <li>• Monitor KPIs</li>
            <li>• Check alarm logs</li>
            <li>• Document changes</li>
          </ul>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
              <Zap className="text-white" size={28} />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Ericsson RAN Features</h1>
              <p className="text-gray-600">Assistant Complet d'Engineering</p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {[
            { id: 'search', label: '🔍 Recherche', icon: Search },
            { id: 'detail', label: '📋 Détails', icon: Settings },
            { id: 'config', label: '⚙️ Configuration', icon: Zap }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-lg font-semibold whitespace-nowrap transition ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-lg shadow-xl p-6">
          {activeTab === 'search' && <SearchTab />}
          {activeTab === 'detail' && <FeatureDetailTab />}
          {activeTab === 'config' && <ConfigurationTab />}
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-sm text-gray-600">
          <p>Base de données : 377 features • 6164 paramètres • 4257 compteurs</p>
        </div>
      </div>
    </div>
  );
};

export default EriccssonRANAssistant;
