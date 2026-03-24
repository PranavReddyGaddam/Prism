import { useState, useCallback } from 'react'
import Matrix from './Matrix'
import classNames from 'classnames'

interface AttentionMatrixProps {
  tokens: string[]
  attentionWeights: number[][]
  layer?: number
  head?: number
  className?: string
  maxTokens?: number
}

export default function AttentionMatrix({
  tokens,
  attentionWeights,
  layer = 0,
  head = 0,
  className,
  maxTokens = 20
}: AttentionMatrixProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null)
  const [highlightRow, setHighlightRow] = useState<number | undefined>()
  const [highlightCol, setHighlightCol] = useState<number | undefined>()

  const displayTokens = tokens.slice(0, maxTokens)
  const displayMatrix = attentionWeights
    .slice(0, maxTokens)
    .map(row => row.slice(0, maxTokens))

  const cleanToken = (token: string) => {
    return token.replace(/^▁/, ' ').replace(/<0x[0-9A-Fa-f]+>/g, '').trim() || token
  }

  const handleCellHover = useCallback((row: number, col: number, value: number) => {
    setHoveredCell({ row, col, value })
    setHighlightRow(row)
    setHighlightCol(col)
  }, [])

  const handleCellLeave = useCallback(() => {
    setHoveredCell(null)
    setHighlightRow(undefined)
    setHighlightCol(undefined)
  }, [])

  const maxVal = Math.max(...displayMatrix.flat(), 0.1)
  
  const colorScale = (value: number) => {
    const normalized = value / maxVal
    return `rgba(139, 92, 246, ${normalized})`
  }

  return (
    <div className={classNames('flex flex-col gap-4', className)}>
      <div className="flex items-center justify-between">
        <div className="text-gray-400 text-xs">
          Layer {layer} · Head {head} · {displayTokens.length} tokens
        </div>
        {hoveredCell && (
          <div className="text-gray-300 text-xs bg-gray-800 px-3 py-1 rounded border border-gray-700">
            <span className="text-purple-400 font-mono">
              {cleanToken(displayTokens[hoveredCell.row])}
            </span>
            {' → '}
            <span className="text-purple-400 font-mono">
              {cleanToken(displayTokens[hoveredCell.col])}
            </span>
            {': '}
            <span className="text-white font-semibold">
              {hoveredCell.value.toFixed(3)}
            </span>
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Column labels */}
          <div className="flex mb-1" style={{ marginLeft: 64 }}>
            {displayTokens.map((token, i) => (
              <div
                key={i}
                className="text-gray-500 text-[9px] text-center truncate"
                style={{
                  width: 28,
                  height: 60,
                  writingMode: 'vertical-rl',
                  paddingBottom: 4,
                  transform: highlightCol === i ? 'scale(1.1)' : 'scale(1)',
                  transition: 'transform 0.15s',
                  color: highlightCol === i ? '#ffffff' : '#9ca3af'
                }}
              >
                {cleanToken(token)}
              </div>
            ))}
          </div>

          {/* Matrix with row labels */}
          <div className="flex">
            {/* Row labels */}
            <div className="flex flex-col justify-around" style={{ width: 60 }}>
              {displayTokens.map((token, i) => (
                <div
                  key={i}
                  className="text-gray-400 text-[9px] text-right pr-2 truncate"
                  style={{
                    height: 28,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    transform: highlightRow === i ? 'scale(1.1)' : 'scale(1)',
                    transition: 'transform 0.15s',
                    color: highlightRow === i ? '#ffffff' : '#9ca3af'
                  }}
                >
                  {cleanToken(token)}
                </div>
              ))}
            </div>

            {/* Matrix */}
            <Matrix
              data={displayMatrix}
              cellHeight={28}
              cellWidth={28}
              rowGap={0}
              colGap={0}
              colorScale={colorScale}
              onCellHover={handleCellHover}
              onCellLeave={handleCellLeave}
              highlightRow={highlightRow}
              highlightCol={highlightCol}
            />
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <span>Attention:</span>
        <div className="flex items-center gap-1">
          <div className="w-8 h-3 rounded" style={{ background: 'rgba(139, 92, 246, 0.2)' }} />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-3 rounded" style={{ background: 'rgba(139, 92, 246, 1)' }} />
          <span>High</span>
        </div>
      </div>
    </div>
  )
}
