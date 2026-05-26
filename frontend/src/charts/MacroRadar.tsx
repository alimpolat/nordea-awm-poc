/**
 * MacroRadar — spider/radar chart of composite macro signals.
 * Single series, clay area fill (semi-transparent).
 */
import ReactECharts from "echarts-for-react";
import { houseChartBase, CLAY, SLATE, GRAY_300, GRAY_500, MONO, SERIF } from "./theme";

export interface RadarIndicator {
  name: string;
  max: number;
}

interface Props {
  indicators: RadarIndicator[];
  values: number[];
  height?: number;
}

export default function MacroRadar({ indicators, values, height = 320 }: Props) {
  const option = {
    ...houseChartBase,
    radar: {
      indicator: indicators,
      shape: "polygon" as const,
      splitNumber: 4,
      axisName: {
        color: SLATE,
        fontFamily: MONO,
        fontSize: 11,
      },
      splitLine: {
        lineStyle: { color: GRAY_300, type: "dashed" as const },
      },
      splitArea: {
        show: true,
        areaStyle: {
          color: ["rgba(250,249,245,0.6)", "rgba(227,218,204,0.25)"],
        },
      },
      axisLine: {
        lineStyle: { color: GRAY_300 },
      },
    },
    series: [
      {
        type: "radar" as const,
        data: [
          {
            value: values,
            name: "Current signal",
            areaStyle: { color: `${CLAY}40` }, // 25% opacity
            lineStyle: { color: CLAY, width: 2 },
            itemStyle: { color: CLAY },
            symbol: "circle",
            symbolSize: 5,
          },
        ],
      },
    ],
    tooltip: {
      ...houseChartBase.tooltip,
      formatter: (params: { name: string; value: number[] }) => {
        const rows = indicators
          .map(
            (ind, i) =>
              `<tr><td style="padding-right:12px;font-family:${MONO};font-size:11px;color:${GRAY_500}">${ind.name}</td><td style="font-weight:600">${params.value[i]}</td></tr>`
          )
          .join("");
        return `<span style="font-family:${SERIF};font-size:13px"><b>Macro signals</b></span><table style="margin-top:6px">${rows}</table>`;
      },
    },
  };

  return (
    <ReactECharts
      option={option}
      style={{ width: "100%", height }}
      opts={{ renderer: "svg" }}
    />
  );
}
