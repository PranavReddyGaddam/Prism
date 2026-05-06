import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { AttributionGraph, AttributionNode, AttributionEdge } from '@/types/attribution'

interface AttributionGraphProps {
  data: AttributionGraph
  width?: number
  height?: number
  onNodeClick?: (node: AttributionNode) => void
  onEdgeClick?: (edge: AttributionEdge) => void
  onInspectNode?: (node: AttributionNode | null) => void
}

export default function AttributionGraphVisualization({
  data,
  width = 1200,
  height = 800,
  onNodeClick,
  onEdgeClick,
  onInspectNode,
}: AttributionGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', '100%')
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')

    // Create container with zoom
    const g = svg.append('g')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Layer-based layout
    const layers = new Map<number, AttributionNode[]>()
    const nodeTypeOrder = { input: 0, intermediate: 1, output: 2, error: 0 }

    data.nodes.forEach(node => {
      const layer = node.layer ?? nodeTypeOrder[node.type]
      if (!layers.has(layer)) layers.set(layer, [])
      layers.get(layer)!.push(node)
    })

    const layerArray = Array.from(layers.entries()).sort((a, b) => a[0] - b[0])
    const layerHeight = height / (layerArray.length + 1)
    const nodePositions = new Map<string, { x: number; y: number }>()

    // Position nodes by layer
    layerArray.forEach(([, nodes], i) => {
      const y = (i + 1) * layerHeight
      const nodeWidth = width / (nodes.length + 1)
      
      nodes.forEach((node, j) => {
        const x = (j + 1) * nodeWidth
        nodePositions.set(node.id, { x, y })
      })
    })

    const nodeColors = {
      input: 'hsl(0 0% 90%)',        // near-white
      intermediate: 'hsl(0 0% 65%)', // mid gray
      output: 'hsl(0 0% 100%)',      // white
      error: 'hsl(0 0% 40%)',        // dark gray
    }

    // Edge weight scale for opacity
    const maxWeight = Math.max(...data.edges.map(e => Math.abs(e.weight)))
    const edgeOpacity = d3.scaleLinear()
      .domain([0, maxWeight])
      .range([0.1, 0.8])

    // Draw edges
    const edges = g.append('g')
      .attr('class', 'edges')
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('x1', d => nodePositions.get(d.source)?.x ?? 0)
      .attr('y1', d => nodePositions.get(d.source)?.y ?? 0)
      .attr('x2', d => nodePositions.get(d.target)?.x ?? 0)
      .attr('y2', d => nodePositions.get(d.target)?.y ?? 0)
      .attr('stroke', 'hsl(0 0% 70%)')
      .attr('stroke-width', d => Math.min(Math.abs(d.weight) / maxWeight * 4, 4))
      .attr('stroke-opacity', d => edgeOpacity(Math.abs(d.weight)))
      .attr('stroke-linecap', 'round')
      .attr('marker-end', 'url(#arrowhead)')
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation()
        onEdgeClick?.(d)
      })
      .on('mouseenter', function() {
        d3.select(this).attr('stroke-width', 6)
      })
      .on('mouseleave', function(_event, d) {
        d3.select(this).attr('stroke-width', Math.min(Math.abs(d.weight) / maxWeight * 5, 5))
      })

    // Add arrowhead marker
    svg.append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 20)
      .attr('refY', 5)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', 'hsl(0 0% 70%)')

    // Draw nodes
    const nodes = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(data.nodes)
      .join('g')
      .attr('transform', d => {
        const pos = nodePositions.get(d.id)
        return `translate(${pos?.x ?? 0}, ${pos?.y ?? 0})`
      })
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation()
        // Cmd+click or Ctrl+click for detailed inspection
        if (event.metaKey || event.ctrlKey) {
          onInspectNode?.(d)
        } else {
          setSelectedNode(d.id)
          onNodeClick?.(d)
        }
      })
      .on('mouseenter', (_event, d) => {
        // Highlight connected edges
        edges
          .attr('stroke-opacity', edge => 
            edge.source === d.id || edge.target === d.id 
              ? 1 
              : edgeOpacity(Math.abs(edge.weight)) * 0.2
          )
      })
      .on('mouseleave', () => {
        edges.attr('stroke-opacity', d => edgeOpacity(Math.abs(d.weight)))
      })

    // Node circles with shadcn styling
    nodes.append('circle')
      .attr('r', d => {
        if (d.type === 'output') return 16
        if (d.type === 'input') return 13
        return 11
      })
      .attr('fill', d => nodeColors[d.type])
      .attr('stroke', d => d.id === selectedNode ? 'hsl(0 0% 100%)' : 'hsl(0 0% 14.9%)')
      .attr('stroke-width', d => d.id === selectedNode ? 3 : 1.5)
      .attr('opacity', d => {
        if (d.type === 'intermediate' && typeof d.activation === 'number') {
          return 0.4 + (d.activation * 0.6)
        }
        return 0.95
      })
      .style('filter', 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3))')

    // Node labels with better typography
    nodes.append('text')
      .attr('dy', -20)
      .attr('text-anchor', 'middle')
      .attr('fill', 'hsl(0 0% 98%)')
      .attr('font-size', '11px')
      .attr('font-weight', d => d.type === 'output' ? '600' : '500')
      .attr('letter-spacing', '0.3px')
      .style('text-shadow', '0 1px 2px rgba(0, 0, 0, 0.5)')
      .text(d => {
        if (d.label.length > 15) return d.label.substring(0, 12) + '...'
        return d.label
      })

    // Probability labels for output nodes
    nodes.filter(d => d.type === 'output' && typeof d.probability === 'number')
      .append('text')
      .attr('dy', 27)
      .attr('text-anchor', 'middle')
      .attr('fill', 'hsl(0 0% 60%)')
      .attr('font-size', '10px')
      .attr('font-weight', '600')
      .style('text-shadow', '0 1px 2px rgba(0, 0, 0, 0.5)')
      .text(d => `${(d.probability! * 100).toFixed(1)}%`)

  }, [data, width, height, selectedNode, onNodeClick, onEdgeClick, onInspectNode])

  return (
    <div className="relative">
      <svg ref={svgRef} className="w-full border border-border rounded-xl bg-background/40 backdrop-blur-sm shadow-lg" />
      
      {/* Legend - collapsed icon, expands on hover */}
      <div className="absolute top-4 right-4 group">
        {/* Icon trigger */}
        <div className="w-7 h-7 rounded-lg bg-card/80 border border-border backdrop-blur-sm shadow flex items-center justify-center cursor-default">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-muted-foreground">
            <circle cx="4" cy="4" r="2" fill="hsl(0 0% 90%)"/>
            <circle cx="4" cy="8" r="2" fill="hsl(0 0% 65%)"/>
            <circle cx="4" cy="12" r="2" fill="hsl(0 0% 40%)"/>
            <rect x="8" y="3" width="6" height="1.5" rx="0.75" fill="currentColor" opacity="0.4"/>
            <rect x="8" y="7" width="6" height="1.5" rx="0.75" fill="currentColor" opacity="0.4"/>
            <rect x="8" y="11" width="6" height="1.5" rx="0.75" fill="currentColor" opacity="0.4"/>
          </svg>
        </div>

        {/* Expanded panel */}
        <div className="absolute top-0 right-0 origin-top-right scale-95 opacity-0 pointer-events-none group-hover:scale-100 group-hover:opacity-100 group-hover:pointer-events-auto transition-all duration-150 ease-out bg-card/95 border border-border rounded-xl p-4 text-xs shadow-lg backdrop-blur-sm w-48">
          <div className="font-semibold text-foreground mb-3 text-sm">Node Types</div>
          <div className="space-y-2">
            <div className="flex items-center gap-2.5">
              <div className="w-3.5 h-3.5 rounded-full shrink-0" style={{backgroundColor: 'hsl(0 0% 90%)'}}></div>
              <span className="text-muted-foreground text-xs">Input (Embeddings)</span>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="w-3.5 h-3.5 rounded-full shrink-0" style={{backgroundColor: 'hsl(0 0% 65%)'}}></div>
              <span className="text-muted-foreground text-xs">Intermediate (Features)</span>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="w-3.5 h-3.5 rounded-full shrink-0" style={{backgroundColor: 'hsl(0 0% 100%)'}}></div>
              <span className="text-muted-foreground text-xs">Output (Tokens)</span>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="w-3.5 h-3.5 rounded-full shrink-0" style={{backgroundColor: 'hsl(0 0% 40%)'}}></div>
              <span className="text-muted-foreground text-xs">Error (Unexplained)</span>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-border space-y-1">
            <div className="text-muted-foreground text-[10px] font-medium">
              Nodes: <span className="text-foreground">{data.nodes.length}</span> / {data.totalNodes}
            </div>
            <div className="text-muted-foreground text-[10px] font-medium">
              Edges: <span className="text-foreground">{data.edges.length}</span> / {data.totalEdges}
            </div>
            {data.explainedBehavior && (
              <div className="text-[10px] mt-1.5 font-semibold text-foreground">
                Explained: {(data.explainedBehavior * 100).toFixed(1)}%
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Selected node info - shadcn styled */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 bg-card/95 border border-border rounded-xl p-4 text-xs max-w-xs shadow-lg backdrop-blur-sm">
          {(() => {
            const node = data.nodes.find(n => n.id === selectedNode)
            if (!node) return null
            return (
              <>
                <div className="font-semibold text-foreground mb-2 text-sm">{node.label}</div>
                <div className="text-muted-foreground text-[11px] space-y-1">
                  <div className="flex justify-between"><span>Type:</span> <span className="text-foreground font-medium">{node.type}</span></div>
                  {typeof node.layer === 'number' && <div className="flex justify-between"><span>Layer:</span> <span className="text-foreground font-medium">{node.layer}</span></div>}
                  {typeof node.position === 'number' && <div className="flex justify-between"><span>Position:</span> <span className="text-foreground font-medium">{node.position}</span></div>}
                  {typeof node.activation === 'number' && <div className="flex justify-between"><span>Activation:</span> <span className="text-foreground font-medium">{node.activation.toFixed(4)}</span></div>}
                  {typeof node.probability === 'number' && <div className="flex justify-between"><span>Probability:</span> <span className="text-foreground font-medium">{(node.probability * 100).toFixed(2)}%</span></div>}
                </div>
                <div className="mt-3 pt-2 border-t border-border">
                  <p className="text-muted-foreground text-[9px] italic">⌘+Click for detailed inspection</p>
                </div>
              </>
            )
          })()}
        </div>
      )}

    </div>
  )
}
