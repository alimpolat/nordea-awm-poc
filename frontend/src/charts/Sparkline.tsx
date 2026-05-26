/**
 * Sparkline — tiny inline line chart for 30-day price trend.
 * No axes, no grid, no tooltip — pure signal.
 * Line colour: olive if positive trend, rust if negative.
 */
import ReactECharts from "echarts-for-react";
import { houseChartBase, OLIVE, RUST } from "./theme";

interface Props {
  data: number[];
  positive?: boolean;
  width?: number;
  height?: number;
}

export default function Sparkline({ data, positive = true, width = 90, height = 28 }: Props) {
  const lineColor = positive ? OLIVE : RUST;

  const option = {
    ...houseChartBase,
    animation: true,
    animationDuration: 600,
    grid: { top: 2, bottom: 2, left: 2, right: 2 },
    xAxis: {
      type: "category" as const,
      show: false,
      boundaryGap: false,
      data: data.map((_, i) => i),
    },
    yAxis: {
      type: "value" as const,
      show: false,
      scale: true,
    },
    series: [
      {
        type: "line" as const,
        data,
        smooth: true,
        symbol: "none",
        lineStyle: { color: lineColor, width: 1.5 },
        areaStyle: {
          color: {
            type: "linear" as const,
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: `${lineColor}40` },
              { offset: 1, color: `${lineColor}00` },
            ],
          },
        },
        itemStyle: { color: lineColor },
      },
    ],
    tooltip: { show: false },
  };

  return (
    <ReactECharts
      option={option}
      style={{ width, height, display: "inline-block" }}
      opts={{ renderer: "svg" }}
    />
  );
}
