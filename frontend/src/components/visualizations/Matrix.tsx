import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import classNames from 'classnames'

interface MatrixProps {
  data: number[][]
  cellHeight?: number
  cellWidth?: number
  rowGap?: number
  colGap?: number
  shape?: 'circle' | 'rect'
  colorScale?: (value: number) => string
  className?: string
  title?: string
  onCellHover?: (row: number, col: number, value: number) => void
  onCellLeave?: () => void
  highlightRow?: number
  highlightCol?: number
  showValues?: boolean
}

export default function Matrix({
  data,
  cellHeight = 28,
  cellWidth = 28,
  rowGap = 2,
  colGap = 2,
  shape = 'rect',
  colorScale,
  className,
  title,
  onCellHover,
  onCellLeave,
  highlightRow,
  highlightCol,
  showValues = false
}: MatrixProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  const rows = data.length
  const cols = data[0]?.length || 0

  const width = cols * (cellWidth + colGap) - colGap
  const height = rows * (cellHeight + rowGap) - rowGap

  const defaultColorScale = (value: number) => {
    const intensity = Math.min(Math.max(value, 0), 1)
    return `rgba(139, 92, 246, ${intensity})`
  }

  const getColor = colorScale || defaultColorScale

  useEffect(() => {
    if (!svgRef.current || !data.length) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const g = svg.append('g')

    data.forEach((row, i) => {
      row.forEach((value, j) => {
        const x = j * (cellWidth + colGap)
        const y = i * (cellHeight + rowGap)

        const isHighlighted = 
          (highlightRow !== undefined && highlightRow === i) ||
          (highlightCol !== undefined && highlightCol === j)

        const cell = g.append(shape === 'circle' ? 'circle' : 'rect')

        if (shape === 'circle') {
          cell
            .attr('cx', x + cellWidth / 2)
            .attr('cy', y + cellHeight / 2)
            .attr('r', Math.min(cellWidth, cellHeight) / 2 - 1)
        } else {
          cell
            .attr('x', x)
            .attr('y', y)
            .attr('width', cellWidth)
            .attr('height', cellHeight)
            .attr('rx', 2)
        }

        cell
          .attr('fill', getColor(value))
          .attr('stroke', isHighlighted ? '#ffffff' : '#1f1f1f')
          .attr('stroke-width', isHighlighted ? 2 : 1)
          .attr('opacity', value === 0 ? 0.1 : 1)
          .style('cursor', 'pointer')
          .on('mouseenter', function() {
            d3.select(this)
              .transition()
              .duration(150)
              .attr('stroke', '#ffffff')
              .attr('stroke-width', 2)
            
            if (onCellHover) {
              onCellHover(i, j, value)
            }
          })
          .on('mouseleave', function() {
            if (!isHighlighted) {
              d3.select(this)
                .transition()
                .duration(150)
                .attr('stroke', '#1f1f1f')
                .attr('stroke-width', 1)
            }
            
            if (onCellLeave) {
              onCellLeave()
            }
          })

        if (showValues && value > 0.01) {
          g.append('text')
            .attr('x', x + cellWidth / 2)
            .attr('y', y + cellHeight / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', value > 0.5 ? '#000' : '#fff')
            .attr('font-size', '9px')
            .attr('pointer-events', 'none')
            .text(value.toFixed(2))
        }
      })
    })
  }, [data, cellHeight, cellWidth, rowGap, colGap, shape, colorScale, highlightRow, highlightCol, showValues, onCellHover, onCellLeave])

  return (
    <div className={classNames('flex flex-col', className)}>
      {title && (
        <h3 className="text-gray-400 text-xs mb-2">{title}</h3>
      )}
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="overflow-visible"
      />
    </div>
  )
}
