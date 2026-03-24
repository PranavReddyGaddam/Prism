import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import classNames from 'classnames'

interface ContinuityFlowProps {
  prompt: string
  result: any
  explainData: any
  className?: string
}

export default function ContinuityFlow({ prompt, result, explainData, className }: ContinuityFlowProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredStage, setHoveredStage] = useState<string | null>(null)

  const stages = [
    { id: 'input', label: 'Input Prompt', color: '#8b5cf6' },
    { id: 'generation', label: 'Model Generation', color: '#06b6d4' },
    { id: 'confidence', label: 'Token Confidence', color: '#22c55e' },
    { id: 'attention', label: 'Attention Patterns', color: '#f59e0b' },
    { id: 'logit', label: 'Logit Lens', color: '#ec4899' },
    { id: 'attribution', label: 'Gradient Attribution', color: '#f97316' },
    { id: 'hidden', label: 'Hidden States', color: '#3b82f6' },
    { id: 'output', label: 'Final Output', color: '#10b981' }
  ]

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !prompt || !result) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = 600

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    svg.attr('width', width).attr('height', height)

    const g = svg.append('g')

    // Define gradients for connections
    const defs = svg.append('defs')
    
    stages.forEach((stage, i) => {
      if (i < stages.length - 1) {
        const gradient = defs.append('linearGradient')
          .attr('id', `flow-gradient-${stage.id}`)
          .attr('x1', '0%')
          .attr('x2', '100%')

        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', stage.color)
          .attr('stop-opacity', 0.8)

        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', stages[i + 1].color)
          .attr('stop-opacity', 0.8)
      }
    })

    // Calculate stage positions
    const stageWidth = 120
    const stageHeight = 80
    const spacing = (width - stages.length * stageWidth) / (stages.length + 1)
    const y = height / 2 - stageHeight / 2

    const stagePositions = stages.map((stage, i) => ({
      ...stage,
      x: spacing + i * (stageWidth + spacing),
      y,
      width: stageWidth,
      height: stageHeight
    }))

    // Draw connections between stages
    stagePositions.forEach((stage, i) => {
      if (i < stagePositions.length - 1) {
        const nextStage = stagePositions[i + 1]
        const x1 = stage.x + stage.width
        const y1 = stage.y + stage.height / 2
        const x2 = nextStage.x
        const y2 = nextStage.y + nextStage.height / 2

        // Create flowing connection path
        const path = d3.path()
        path.moveTo(x1, y1)
        
        const controlX1 = x1 + (x2 - x1) / 3
        const controlX2 = x1 + 2 * (x2 - x1) / 3
        
        path.bezierCurveTo(
          controlX1, y1,
          controlX2, y2,
          x2, y2
        )

        g.append('path')
          .attr('d', path.toString())
          .attr('stroke', `url(#flow-gradient-${stage.id})`)
          .attr('stroke-width', 3)
          .attr('fill', 'none')
          .attr('opacity', hoveredStage === stage.id || hoveredStage === stages[i + 1].id ? 1 : 0.4)
          .style('transition', 'all 0.3s')
          .on('mouseenter', () => setHoveredStage(stage.id))
          .on('mouseleave', () => setHoveredStage(null))

        // Add animated dots along the path
        const dots = g.append('circle')
          .attr('r', 4)
          .attr('fill', stage.color)
          .attr('opacity', 0.8)

        // Animate dots along path
        function animateDot() {
          const pathNode = g.select('path').node() as SVGPathElement
          if (!pathNode) return
          
          const pathLength = pathNode.getTotalLength()
          
          dots
            .transition()
            .duration(3000)
            .ease(d3.easeLinear)
            .attrTween('transform', () => {
              return (t: number) => {
                const point = pathNode.getPointAtLength(t * pathLength)
                return `translate(${point.x}, ${point.y})`
              }
            })
            .on('end', animateDot)
        }
        
        setTimeout(animateDot, i * 500)
      }
    })

    // Draw stage boxes
    stagePositions.forEach((stage) => {
      const stageGroup = g.append('g')
        .style('cursor', 'pointer')
        .on('mouseenter', () => setHoveredStage(stage.id))
        .on('mouseleave', () => setHoveredStage(null))

      // Stage box
      stageGroup.append('rect')
        .attr('x', stage.x)
        .attr('y', stage.y)
        .attr('width', stage.width)
        .attr('height', stage.height)
        .attr('rx', 8)
        .attr('fill', hoveredStage === stage.id ? stage.color : '#1f1f1f')
        .attr('stroke', stage.color)
        .attr('stroke-width', hoveredStage === stage.id ? 3 : 2)
        .style('transition', 'all 0.3s')

      // Stage label
      stageGroup.append('text')
        .attr('x', stage.x + stage.width / 2)
        .attr('y', stage.y + stage.height / 2 - 10)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', hoveredStage === stage.id ? '#000' : '#fff')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text(stage.label)

      // Stage status indicator
      const hasData = getStageData(stage.id, prompt, result, explainData)
      stageGroup.append('circle')
        .attr('cx', stage.x + stage.width - 10)
        .attr('cy', stage.y + 10)
        .attr('r', 5)
        .attr('fill', hasData ? '#22c55e' : '#6b7280')
        .attr('opacity', 0.8)

      // Add stage-specific content preview
      if (hasData) {
        const preview = getStagePreview(stage.id, prompt, result, explainData)
        stageGroup.append('text')
          .attr('x', stage.x + stage.width / 2)
          .attr('y', stage.y + stage.height / 2 + 10)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', hoveredStage === stage.id ? '#000' : '#9ca3af')
          .attr('font-size', '9px')
          .text(preview)
      }
    })

  }, [prompt, result, explainData, hoveredStage])

  const getStageData = (stageId: string, prompt: string, result: any, explainData: any) => {
    switch (stageId) {
      case 'input': return !!prompt
      case 'generation': return !!result?.response
      case 'confidence': return !!explainData?.confidence?.length
      case 'attention': return !!explainData?.attention
      case 'logit': return !!explainData?.logitLens?.length
      case 'attribution': return !!explainData?.attribution?.length
      case 'hidden': return !!explainData?.hiddenStates?.length
      case 'output': return !!result?.final_answer
      default: return false
    }
  }

  const getStagePreview = (stageId: string, prompt: string, result: any, explainData: any) => {
    switch (stageId) {
      case 'input': return prompt.substring(0, 15) + '...'
      case 'generation': return `${result?.token_count || 0} tokens`
      case 'confidence': return `${explainData?.confidence?.length || 0} tokens`
      case 'attention': return `${explainData?.attention?.layer || 0}:${explainData?.attention?.head || 0}`
      case 'logit': return `${explainData?.logitLens?.length || 0} layers`
      case 'attribution': return `${explainData?.attribution?.length || 0} scores`
      case 'hidden': return `${explainData?.hiddenStates?.length || 0} states`
      case 'output': return result?.final_answer?.substring(0, 15) + '...' || 'Ready'
      default: return ''
    }
  }

  return (
    <div ref={containerRef} className={classNames('w-full bg-gray-900 rounded-lg p-4', className)}>
      <h3 className="text-white text-lg font-semibold mb-4">Model Processing Flow</h3>
      <svg ref={svgRef} className="w-full" />
      <div className="mt-4 flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Data Available</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-500" />
            <span>Processing</span>
          </div>
        </div>
        <div className="text-gray-400">
          Hover over stages to highlight connections
        </div>
      </div>
    </div>
  )
}
