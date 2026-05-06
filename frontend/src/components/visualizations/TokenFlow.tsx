import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface TokenFlowProps {
  tokens: string[]
  confidence: number[]
  className?: string
}

export default function TokenFlow({ tokens, confidence, className }: TokenFlowProps) {
  const cleanToken = (token: string) => {
    return token.replace(/^▁/, ' ').replace(/<0x[0-9A-Fa-f]+>/g, '').trim() || token
  }

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return 'hsl(142.1 76.2% 45%)' // green
    if (conf >= 0.6) return 'hsl(221.2 83.2% 53.3%)' // blue
    if (conf >= 0.4) return 'hsl(45 93.4% 47.5%)' // yellow
    return 'hsl(0 84.2% 60.2%)' // red
  }

  const confidenceValues = confidence.map(c => (c || 0) * 100)
  const avgConfidence = confidenceValues.reduce((sum, val) => sum + val, 0) / confidenceValues.length
  const minConfidence = Math.min(...confidenceValues)
  const maxConfidence = Math.max(...confidenceValues)

  const data = tokens.map((token, i) => ({
    token: cleanToken(token),
    confidence: confidenceValues[i],
    position: i + 1,
    fullToken: token
  }))

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const tokenData = payload[0].payload
      const conf = tokenData.confidence / 100
      return (
        <div className="bg-card/95 border border-border rounded-lg p-3 shadow-lg backdrop-blur-sm">
          <p className="text-foreground font-semibold mb-1">Position #{tokenData.position}</p>
          <p className="text-muted-foreground text-sm mb-2">
            <span className="font-mono">{tokenData.token}</span>
          </p>
          <p className="text-sm font-semibold" style={{ color: getConfidenceColor(conf) }}>
            Confidence: {tokenData.confidence.toFixed(1)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {tokenData.confidence > avgConfidence ? '↑ Above' : '↓ Below'} average
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className={className}>
      <div className="mb-4 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <div className="text-muted-foreground">
            Total Tokens: <span className="text-foreground font-semibold">{tokens.length}</span>
          </div>
          <div className="text-muted-foreground">
            Average: <span className="text-foreground font-semibold">{avgConfidence.toFixed(1)}%</span>
          </div>
        </div>
        <div className="flex items-center justify-between text-xs">
          <div className="text-muted-foreground">
            Min: <span className="text-foreground font-semibold">{minConfidence.toFixed(1)}%</span>
          </div>
          <div className="text-muted-foreground">
            Max: <span className="text-foreground font-semibold">{maxConfidence.toFixed(1)}%</span>
          </div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={400}>
        <AreaChart
          data={data}
          margin={{ top: 10, right: 30, left: 10, bottom: 30 }}
        >
          <defs>
            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(221.2 83.2% 53.3%)" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="hsl(221.2 83.2% 53.3%)" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 20%)" />
          <XAxis
            dataKey="position"
            stroke="hsl(0 0% 60%)"
            tick={{ fill: 'hsl(0 0% 70%)', fontSize: 11 }}
            label={{ value: 'Token Position', position: 'insideBottom', offset: -10, fill: 'hsl(0 0% 70%)' }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="hsl(0 0% 60%)"
            tick={{ fill: 'hsl(0 0% 70%)', fontSize: 11 }}
            label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft', fill: 'hsl(0 0% 70%)' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={avgConfidence}
            stroke="hsl(45 93.4% 47.5%)"
            strokeDasharray="5 5"
            strokeWidth={2}
            label={{
              value: `Avg: ${avgConfidence.toFixed(1)}%`,
              position: 'right',
              fill: 'hsl(45 93.4% 47.5%)',
              fontSize: 11,
              fontWeight: 600
            }}
          />
          <Area
            type="monotone"
            dataKey="confidence"
            stroke="hsl(221.2 83.2% 53.3%)"
            strokeWidth={2}
            fill="url(#confidenceGradient)"
            animationDuration={1000}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
