import type { AttributionNode } from '@/types/attribution'

interface NodeDetailPanelProps {
  node: AttributionNode
  onClose: () => void
}

export default function NodeDetailPanel({ node, onClose }: NodeDetailPanelProps) {
  return (
    <div className="fixed right-4 top-20 bottom-4 w-96 bg-gray-900/95 border border-gray-700 rounded-xl p-4 overflow-y-auto backdrop-blur-xl z-50"
      style={{
        backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.4) 0%, rgba(29, 29, 29, 0.6) 50%, rgba(67, 67, 67, 0.4) 100%)'
      }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-700">
        <div>
          <h3 className="text-white font-semibold text-lg">{node.label}</h3>
          <p className="text-gray-400 text-xs mt-1">
            {node.type.charAt(0).toUpperCase() + node.type.slice(1)} Node
            {node.layer !== undefined && ` · Layer ${node.layer}`}
            {node.position !== undefined && ` · Position ${node.position}`}
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors p-1 hover:bg-gray-800 rounded"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Node Info */}
      <div className="space-y-4">
        {/* Activation/Probability */}
        {typeof node.activation === 'number' && (
          <div className="bg-gray-800/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs mb-1">Activation</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                  style={{ width: `${node.activation * 100}%` }}
                />
              </div>
              <span className="text-white text-sm font-semibold">{(node.activation * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}

        {typeof node.probability === 'number' && (
          <div className="bg-gray-800/50 rounded-lg p-3">
            <p className="text-gray-400 text-xs mb-1">Probability</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-green-500 to-emerald-500"
                  style={{ width: `${node.probability * 100}%` }}
                />
              </div>
              <span className="text-white text-sm font-semibold">{(node.probability * 100).toFixed(2)}%</span>
            </div>
          </div>
        )}

        {/* Subgraph Interval */}
        {node.subgraphInterval && (
          <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-3">
            <p className="text-blue-400 text-xs font-semibold">{node.subgraphInterval}</p>
          </div>
        )}

        {/* Input Features */}
        {node.inputFeatures && node.inputFeatures.length > 0 && (
          <div>
            <h4 className="text-white text-sm font-semibold mb-2 flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
              Input Features
            </h4>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {node.inputFeatures.map((feat, i) => (
                <div key={i} className="bg-gray-800/50 rounded p-2 text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-gray-300 font-mono">{feat.feature}</span>
                    <span className={`font-semibold ${feat.weight > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(3)}
                    </span>
                  </div>
                  {feat.description && (
                    <p className="text-gray-500 text-[10px]">{feat.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Output Features */}
        {node.outputFeatures && node.outputFeatures.length > 0 && (
          <div>
            <h4 className="text-white text-sm font-semibold mb-2 flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500"></div>
              Output Features
            </h4>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {node.outputFeatures.map((feat, i) => (
                <div key={i} className="bg-gray-800/50 rounded p-2 text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-gray-300 font-mono">{feat.feature}</span>
                    <span className={`font-semibold ${feat.weight > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(3)}
                    </span>
                  </div>
                  {feat.description && (
                    <p className="text-gray-500 text-[10px]">{feat.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Top Activations */}
        {node.topActivations && node.topActivations.length > 0 && (
          <div>
            <h4 className="text-white text-sm font-semibold mb-2 flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-yellow-500"></div>
              Top Activations
            </h4>
            <div className="space-y-1">
              {node.topActivations.map((activation, i) => (
                <div key={i} className="bg-gray-800/50 rounded p-2">
                  <p className="text-gray-300 text-xs font-mono">{activation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Token Predictions */}
        {node.tokenPredictions && node.tokenPredictions.length > 0 && (
          <div>
            <h4 className="text-white text-sm font-semibold mb-2 flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-500"></div>
              Token Predictions
            </h4>
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {node.tokenPredictions.map((pred, i) => (
                <div key={i} className="bg-gray-800/50 rounded p-2 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-gray-500 w-6 text-right">{pred.rank}</span>
                    <span className="text-gray-300 font-mono truncate">{pred.token}</span>
                  </div>
                  <div className="flex items-center gap-2 ml-2">
                    <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                        style={{ width: `${pred.probability * 100}%` }}
                      />
                    </div>
                    <span className="text-white font-semibold w-12 text-right">
                      {(pred.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
