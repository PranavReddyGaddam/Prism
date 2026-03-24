import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import classNames from 'classnames'

interface BeautifulFlowProps {
  prompt: string
  result: any
  explainData: any
  className?: string
}

export default function BeautifulFlow({ prompt, result, explainData, className }: BeautifulFlowProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [activeStage, setActiveStage] = useState<number>(0)

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !prompt) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = 1400

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    svg.attr('width', width).attr('height', height)

    const g = svg.append('g')

    // Define beautiful gradients
    const defs = svg.append('defs')
    
    // Main flow gradient
    const flowGradient = defs.append('linearGradient')
      .attr('id', 'flow-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%')

    flowGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#8b5cf6')
      .attr('stop-opacity', 0.8)

    flowGradient.append('stop')
      .attr('offset', '50%')
      .attr('stop-color', '#06b6d4')
      .attr('stop-opacity', 0.8)

    flowGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#10b981')
      .attr('stop-opacity', 0.8)

    // Glow filter
    const glow = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%')

    glow.append('feGaussianBlur')
      .attr('stdDeviation', '4')
      .attr('result', 'coloredBlur')

    const feMerge = glow.append('feMerge')
    feMerge.append('feMergeNode').attr('in', 'coloredBlur')
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic')

    const centerX = width / 2
    const stageSpacing = 180

    // Stage 1: Input Prompt
    const stage1Y = 50
    drawStage(g, centerX, stage1Y, 'Your Question', prompt.substring(0, 50) + '...', '#8b5cf6', 0)
    
    // Tokens visualization
    const tokens = explainData?.confidence?.slice(0, 8) || []
    if (tokens.length > 0) {
      drawTokens(g, centerX, stage1Y + 80, tokens, 1)
    }

    // Flow arrow 1
    drawFlowArrow(g, centerX, stage1Y + 120, stage1Y + stageSpacing - 20, '#8b5cf6', '#06b6d4')

    // Stage 2: Model Processing
    const stage2Y = stage1Y + stageSpacing
    drawStage(g, centerX, stage2Y, 'AI Model Thinks', `Processing ${result?.token_count || 0} tokens`, '#06b6d4', 1)
    
    // Neural network visualization
    drawNeuralNetwork(g, centerX, stage2Y + 80, 2)

    // Flow arrow 2
    drawFlowArrow(g, centerX, stage2Y + 140, stage2Y + stageSpacing - 20, '#06b6d4', '#f59e0b')

    // Stage 3: Attention Analysis
    const stage3Y = stage2Y + stageSpacing
    drawStage(g, centerX, stage3Y, 'Attention Patterns', 'Which words matter most?', '#f59e0b', 2)
    
    // Mini attention heatmap
    if (explainData?.attention?.matrix) {
      drawMiniAttention(g, centerX, stage3Y + 80, explainData.attention.matrix.slice(0, 6), 3)
    }

    // Flow arrow 3
    drawFlowArrow(g, centerX, stage3Y + 140, stage3Y + stageSpacing - 20, '#f59e0b', '#ec4899')

    // Stage 4: Confidence Check
    const stage4Y = stage3Y + stageSpacing
    drawStage(g, centerX, stage4Y, 'Confidence Analysis', 'How sure is the AI?', '#ec4899', 3)
    
    // Confidence bars
    if (tokens.length > 0) {
      drawConfidenceBars(g, centerX, stage4Y + 80, tokens.slice(0, 5), 4)
    }

    // Flow arrow 4
    drawFlowArrow(g, centerX, stage4Y + 140, stage4Y + stageSpacing - 20, '#ec4899', '#10b981')

    // Stage 5: Final Answer
    const stage5Y = stage4Y + stageSpacing
    drawStage(g, centerX, stage5Y, 'Final Answer', result?.final_answer?.substring(0, 50) + '...' || 'Ready', '#10b981', 4)

    // Helper functions
    function drawStage(parent: any, x: number, y: number, title: string, subtitle: string, color: string, index: number) {
      const stageGroup = parent.append('g')
        .style('cursor', 'pointer')
        .on('mouseenter', () => setActiveStage(index))
        .on('mouseleave', () => setActiveStage(-1))

      // Outer glow circle
      stageGroup.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 65)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('opacity', activeStage === index ? 0.6 : 0.2)
        .attr('filter', 'url(#glow)')
        .transition()
        .duration(300)

      // Main circle
      stageGroup.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 50)
        .attr('fill', activeStage === index ? color : '#1f1f1f')
        .attr('stroke', color)
        .attr('stroke-width', 3)
        .transition()
        .duration(300)

      // Title
      stageGroup.append('text')
        .attr('x', x)
        .attr('y', y - 10)
        .attr('text-anchor', 'middle')
        .attr('fill', activeStage === index ? '#000' : '#fff')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text(title)

      // Subtitle
      stageGroup.append('text')
        .attr('x', x)
        .attr('y', y + 10)
        .attr('text-anchor', 'middle')
        .attr('fill', activeStage === index ? '#000' : '#9ca3af')
        .attr('font-size', '10px')
        .text(subtitle.length > 30 ? subtitle.substring(0, 30) + '...' : subtitle)

      // Stage number badge
      stageGroup.append('circle')
        .attr('cx', x + 40)
        .attr('cy', y - 40)
        .attr('r', 15)
        .attr('fill', color)

      stageGroup.append('text')
        .attr('x', x + 40)
        .attr('y', y - 40)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#000')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .text(index + 1)
    }

    function drawFlowArrow(parent: any, x: number, y1: number, y2: number, color1: string, color2: string) {
      const gradient = defs.append('linearGradient')
        .attr('id', `arrow-gradient-${y1}`)
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '0%')
        .attr('y2', '100%')

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', color1)

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', color2)

      // Main flow line
      parent.append('line')
        .attr('x1', x)
        .attr('y1', y1)
        .attr('x2', x)
        .attr('y2', y2)
        .attr('stroke', `url(#arrow-gradient-${y1})`)
        .attr('stroke-width', 4)
        .attr('opacity', 0.8)

      // Animated dots
      for (let i = 0; i < 3; i++) {
        const dot = parent.append('circle')
          .attr('cx', x)
          .attr('cy', y1)
          .attr('r', 4)
          .attr('fill', color1)
          .attr('opacity', 0.8)

        function animateDot() {
          dot
            .attr('cy', y1)
            .transition()
            .duration(2000)
            .delay(i * 700)
            .attr('cy', y2)
            .attr('fill', color2)
            .on('end', animateDot)
        }
        animateDot()
      }

      // Arrow head
      parent.append('polygon')
        .attr('points', `${x},${y2} ${x - 6},${y2 - 10} ${x + 6},${y2 - 10}`)
        .attr('fill', color2)
    }

    function drawTokens(parent: any, x: number, y: number, tokens: any[], stageIndex: number) {
      const tokenGroup = parent.append('g')
        .attr('opacity', activeStage === stageIndex ? 1 : 0.6)

      tokens.forEach((token, i) => {
        const tokenX = x - (tokens.length * 35) / 2 + i * 35
        
        tokenGroup.append('rect')
          .attr('x', tokenX)
          .attr('y', y)
          .attr('width', 30)
          .attr('height', 20)
          .attr('rx', 4)
          .attr('fill', d3.interpolateRgb('#ef4444', '#22c55e')(token.confidence))
          .attr('opacity', 0.8)

        tokenGroup.append('text')
          .attr('x', tokenX + 15)
          .attr('y', y + 13)
          .attr('text-anchor', 'middle')
          .attr('fill', '#000')
          .attr('font-size', '8px')
          .attr('font-weight', 'bold')
          .text(token.token.substring(0, 3))
      })
    }

    function drawNeuralNetwork(parent: any, x: number, y: number, stageIndex: number) {
      const networkGroup = parent.append('g')
        .attr('opacity', activeStage === stageIndex ? 1 : 0.6)

      const layers = [4, 6, 4]
      const layerSpacing = 40
      const nodeSpacing = 20

      layers.forEach((nodeCount, layerIndex) => {
        const layerX = x - (layers.length * layerSpacing) / 2 + layerIndex * layerSpacing
        
        for (let i = 0; i < nodeCount; i++) {
          const nodeY = y - (nodeCount * nodeSpacing) / 2 + i * nodeSpacing
          
          networkGroup.append('circle')
            .attr('cx', layerX)
            .attr('cy', nodeY)
            .attr('r', 4)
            .attr('fill', '#06b6d4')
            .attr('opacity', 0.8)

          // Connect to next layer
          if (layerIndex < layers.length - 1) {
            const nextLayerX = layerX + layerSpacing
            const nextNodeCount = layers[layerIndex + 1]
            
            for (let j = 0; j < nextNodeCount; j++) {
              const nextNodeY = y - (nextNodeCount * nodeSpacing) / 2 + j * nodeSpacing
              
              networkGroup.append('line')
                .attr('x1', layerX)
                .attr('y1', nodeY)
                .attr('x2', nextLayerX)
                .attr('y2', nextNodeY)
                .attr('stroke', '#06b6d4')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.3)
            }
          }
        }
      })
    }

    function drawMiniAttention(parent: any, x: number, y: number, matrix: number[][], stageIndex: number) {
      const attentionGroup = parent.append('g')
        .attr('opacity', activeStage === stageIndex ? 1 : 0.6)

      const cellSize = 12
      const startX = x - (matrix.length * cellSize) / 2

      matrix.forEach((row, i) => {
        row.slice(0, 6).forEach((value, j) => {
          attentionGroup.append('rect')
            .attr('x', startX + j * cellSize)
            .attr('y', y + i * cellSize)
            .attr('width', cellSize - 1)
            .attr('height', cellSize - 1)
            .attr('fill', `rgba(245, 158, 11, ${value})`)
            .attr('rx', 1)
        })
      })
    }

    function drawConfidenceBars(parent: any, x: number, y: number, tokens: any[], stageIndex: number) {
      const barsGroup = parent.append('g')
        .attr('opacity', activeStage === stageIndex ? 1 : 0.6)

      const barWidth = 60
      const barHeight = 8
      const barSpacing = 12

      tokens.forEach((token, i) => {
        const barY = y + i * barSpacing
        
        // Background
        barsGroup.append('rect')
          .attr('x', x - barWidth / 2)
          .attr('y', barY)
          .attr('width', barWidth)
          .attr('height', barHeight)
          .attr('fill', '#1f1f1f')
          .attr('rx', 2)

        // Confidence bar
        barsGroup.append('rect')
          .attr('x', x - barWidth / 2)
          .attr('y', barY)
          .attr('width', 0)
          .attr('height', barHeight)
          .attr('fill', d3.interpolateRgb('#ef4444', '#22c55e')(token.confidence))
          .attr('rx', 2)
          .transition()
          .duration(1000)
          .delay(i * 100)
          .attr('width', barWidth * token.confidence)
      })
    }

  }, [prompt, result, explainData, activeStage])

  return (
    <div ref={containerRef} className={classNames('w-full bg-gradient-to-b from-gray-900 to-black rounded-lg p-8', className)}>
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">How AI Understands Your Question</h2>
        <p className="text-gray-400 text-sm">Watch your question flow through the AI's mind</p>
      </div>
      <svg ref={svgRef} className="w-full" />
      <div className="mt-8 grid grid-cols-5 gap-4 text-center text-xs">
        <div className="space-y-1">
          <div className="w-8 h-8 rounded-full bg-purple-600 mx-auto" />
          <p className="text-white font-semibold">Your Input</p>
          <p className="text-gray-500">Question received</p>
        </div>
        <div className="space-y-1">
          <div className="w-8 h-8 rounded-full bg-cyan-600 mx-auto" />
          <p className="text-white font-semibold">Processing</p>
          <p className="text-gray-500">AI analyzes</p>
        </div>
        <div className="space-y-1">
          <div className="w-8 h-8 rounded-full bg-amber-600 mx-auto" />
          <p className="text-white font-semibold">Attention</p>
          <p className="text-gray-500">Focus on key words</p>
        </div>
        <div className="space-y-1">
          <div className="w-8 h-8 rounded-full bg-pink-600 mx-auto" />
          <p className="text-white font-semibold">Confidence</p>
          <p className="text-gray-500">Certainty check</p>
        </div>
        <div className="space-y-1">
          <div className="w-8 h-8 rounded-full bg-green-600 mx-auto" />
          <p className="text-white font-semibold">Answer</p>
          <p className="text-gray-500">Final response</p>
        </div>
      </div>
    </div>
  )
}
