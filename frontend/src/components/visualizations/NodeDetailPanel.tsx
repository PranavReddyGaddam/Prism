import type { AttributionNode } from '@/types/attribution'

interface NodeDetailPanelProps {
  node: AttributionNode
  onClose: () => void
}

export default function NodeDetailPanel({ node, onClose }: NodeDetailPanelProps) {
  return (
    <div className="h-full border rounded-xl flex flex-col" style={{backgroundColor: '#202020', borderColor: '#333'}}>
      {/* Header — pinned */}
      <div className="shrink-0 flex items-center justify-between px-4 pt-4 pb-3 border-b" style={{borderColor: '#333'}}>
        <div>
          <h3 className="text-white font-semibold text-lg">{node.label}</h3>
          <p className="text-gray-500 text-xs mt-1">
            {node.type.charAt(0).toUpperCase() + node.type.slice(1)} Node
            {node.layer !== undefined && ` · Layer ${node.layer}`}
            {node.position !== undefined && ` · Position ${node.position}`}
          </p>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-white transition-colors p-1 rounded"
          onMouseEnter={e => (e.currentTarget.style.backgroundColor='#3a3a3a')}
          onMouseLeave={e => (e.currentTarget.style.backgroundColor='')}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {typeof node.activation === 'number' && (
          <div className="rounded-lg p-3" style={{backgroundColor: '#2a2a2a'}}>
            <p className="text-gray-500 text-xs mb-2">Activation</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{backgroundColor: '#3a3a3a'}}>
                <div
                  className="h-full bg-white"
                  style={{ width: `${node.activation * 100}%` }}
                />
              </div>
              <span className="text-white text-sm font-semibold tabular-nums">{(node.activation * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}

        {typeof node.probability === 'number' && (
          <div className="rounded-lg p-3" style={{backgroundColor: '#2a2a2a'}}>
            <p className="text-gray-500 text-xs mb-2">Probability</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{backgroundColor: '#3a3a3a'}}>
                <div
                  className="h-full bg-white"
                  style={{ width: `${node.probability * 100}%` }}
                />
              </div>
              <span className="text-white text-sm font-semibold tabular-nums">{(node.probability * 100).toFixed(2)}%</span>
            </div>
          </div>
        )}

        {node.subgraphInterval && (
          <div className="rounded-lg p-3 border" style={{backgroundColor: '#2a2a2a', borderColor: '#333'}}>
            <p className="text-gray-300 text-xs font-medium">{node.subgraphInterval}</p>
          </div>
        )}

        {node.inputFeatures && node.inputFeatures.length > 0 && (
          <div>
            <h4 className="text-gray-300 text-xs font-semibold uppercase tracking-wider mb-2">Input Features</h4>
            <div className="space-y-1">
              {node.inputFeatures.map((feat, i) => (
                <div key={i} className="rounded p-2 text-xs" style={{backgroundColor: '#2a2a2a'}}>
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-gray-300 font-mono">{feat.feature}</span>
                    <span className={`font-semibold tabular-nums ${feat.weight > 0 ? 'text-white' : 'text-gray-400'}`}>
                      {feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(3)}
                    </span>
                  </div>
                  {feat.description && (
                    <p className="text-gray-400 text-[10px]">{feat.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {node.outputFeatures && node.outputFeatures.length > 0 && (
          <div>
            <h4 className="text-gray-300 text-xs font-semibold uppercase tracking-wider mb-2">Output Features</h4>
            <div className="space-y-1">
              {node.outputFeatures.map((feat, i) => (
                <div key={i} className="rounded p-2 text-xs" style={{backgroundColor: '#2a2a2a'}}>
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-gray-300 font-mono">{feat.feature}</span>
                    <span className={`font-semibold tabular-nums ${feat.weight > 0 ? 'text-white' : 'text-gray-400'}`}>
                      {feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(3)}
                    </span>
                  </div>
                  {feat.description && (
                    <p className="text-gray-400 text-[10px]">{feat.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {node.topActivations && node.topActivations.length > 0 && (
          <div>
            <h4 className="text-gray-300 text-xs font-semibold uppercase tracking-wider mb-2">Top Activations</h4>
            <div className="space-y-1">
              {node.topActivations.map((activation, i) => (
                <div key={i} className="rounded p-2" style={{backgroundColor: '#2a2a2a'}}>
                  <p className="text-gray-300 text-xs font-mono">{activation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {node.tokenPredictions && node.tokenPredictions.length > 0 && (
          <div>
            <h4 className="text-gray-300 text-xs font-semibold uppercase tracking-wider mb-2">Token Predictions</h4>
            <div className="space-y-1">
              {node.tokenPredictions.map((pred, i) => (
                <div key={i} className="rounded p-2 flex items-center justify-between text-xs" style={{backgroundColor: '#2a2a2a'}}>
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span className="text-gray-600 w-6 text-right tabular-nums">{pred.rank}</span>
                    <span className="text-gray-300 font-mono truncate">{pred.token}</span>
                  </div>
                  <div className="flex items-center gap-2 ml-2">
                    <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{backgroundColor: '#3a3a3a'}}>
                      <div
                        className="h-full bg-white"
                        style={{ width: `${pred.probability * 100}%` }}
                      />
                    </div>
                    <span className="text-white font-semibold w-12 text-right tabular-nums">
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
