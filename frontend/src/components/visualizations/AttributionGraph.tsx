import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { AttributionGraph, AttributionNode, AttributionEdge } from '@/types/attribution'
import NodeDetailPanel from './NodeDetailPanel'

interface AttributionGraphProps {
  data: AttributionGraph
  width?: number
  height?: number
  onNodeClick?: (node: AttributionNode) => void
  onEdgeClick?: (edge: AttributionEdge) => void
}

export default function AttributionGraphVisualization({
  data,
  width = 1200,
  height = 800,
  onNodeClick,
  onEdgeClick
}: AttributionGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [inspectedNode, setInspectedNode] = useState<AttributionNode | null>(null)

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

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

    // Monochromatic blue color scheme
    const nodeColors = {
      input: 'hsl(221.2 83.2% 53.3%)',      // bright blue
      intermediate: 'hsl(221.2 70% 45%)',   // medium blue
      output: 'hsl(221.2 60% 38%)',         // dark blue
      error: 'hsl(221.2 50% 30%)'           // darker blue
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
      .attr('stroke', d => d.weight > 0 ? 'hsl(221.2 70% 50%)' : 'hsl(221.2 60% 40%)')
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
      .attr('fill', 'hsl(221.2 70% 50%)')

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
          setInspectedNode(d)
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
      .attr('fill', 'hsl(221.2 70% 55%)')
      .attr('font-size', '10px')
      .attr('font-weight', '600')
      .style('text-shadow', '0 1px 2px rgba(0, 0, 0, 0.5)')
      .text(d => `${(d.probability! * 100).toFixed(1)}%`)

  }, [data, width, height, selectedNode, onNodeClick, onEdgeClick])

  return (
    <div className="relative">
      <svg ref={svgRef} className="border border-border rounded-xl bg-background/40 backdrop-blur-sm shadow-lg" />
      
      {/* Legend - shadcn styled */}
      <div className="absolute top-4 right-4 bg-card/95 border border-border rounded-xl p-4 text-xs shadow-lg backdrop-blur-sm">
        <div className="font-semibold text-foreground mb-3 text-sm">Node Types</div>
        <div className="space-y-2">
          <div className="flex items-center gap-2.5">
            <div className="w-3.5 h-3.5 rounded-full" style={{backgroundColor: 'hsl(221.2 83.2% 53.3%)'}}></div>
            <span className="text-muted-foreground text-xs">Input (Embeddings)</span>
          </div>
          <div className="flex items-center gap-2.5">
            <div className="w-3.5 h-3.5 rounded-full" style={{backgroundColor: 'hsl(221.2 70% 45%)'}}></div>
            <span className="text-muted-foreground text-xs">Intermediate (Features)</span>
          </div>
          <div className="flex items-center gap-2.5">
            <div className="w-3.5 h-3.5 rounded-full" style={{backgroundColor: 'hsl(221.2 60% 38%)'}}></div>
            <span className="text-muted-foreground text-xs">Output (Tokens)</span>
          </div>
          <div className="flex items-center gap-2.5">
            <div className="w-3.5 h-3.5 rounded-full" style={{backgroundColor: 'hsl(221.2 50% 30%)'}}></div>
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
            <div className="text-[10px] mt-1.5 font-semibold" style={{color: 'hsl(221.2 70% 55%)'}}>
              Explained: {(data.explainedBehavior * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </div>

      {/* Selected node info - shadcn styled */}
      {selectedNode && !inspectedNode && (
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

      {/* Node Detail Panel */}
      {inspectedNode && (
        <NodeDetailPanel 
          node={inspectedNode}
          onClose={() => setInspectedNode(null)}
        />
      )}
    </div>
  )
}
