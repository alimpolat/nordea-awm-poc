/**
 * AllocationDonut — two concentric donut rings.
 * Inner ring = target allocation, outer ring = current allocation.
 * Props mirror ClientSnapshot.target_allocation shape.
 */
import ReactECharts from "echarts-for-react";
import { houseChartBase, SERIF, CLAY, OLIVE, AMBER, RUST, GRAY_500, SLATE } from "./theme";

interface Props {
  /** fractions summing ~1, e.g. { "Nordic equity": 0.35, ... } */
  target: Record<string, number>;
  current: Record<string, number>;
  height?: number;
}

const ASSET_COLORS: Record<string, string> = {
  "Nordic equity": CLAY,
  "US tech": OLIVE,
  "EU fixed income": AMBER,
  "Gulf real estate": RUST,
  Alternatives: GRAY_500,
};

function pickColor(name: string, idx: number): string {
  const palette = [CLAY, OLIVE, AMBER, RUST, GRAY_500, "#5B8DB8"];
  return ASSET_COLORS[name] ?? palette[idx % palette.length];
}

export default function AllocationDonut({ target, current, height = 340 }: Props) {
  const assetClasses = Object.keys(target);

  const innerData = assetClasses.map((name, i) => ({
    name,
    value: +(target[name] * 100).toFixed(1),
    itemStyle: { color: pickColor(name, i) },
  }));

  const outerData = assetClasses.map((name, i) => ({
    name,
    value: +(( current[name] ?? 0) * 100).toFixed(1),
    itemStyle: { color: pickColor(name, i), opacity: 0.72 },
  }));

  const option = {
    ...houseChartBase,
    // Legend BELOW the donut (horizontal, wrapping) — a vertical right-side legend
    // collides with the rings inside the narrow (~340px) allocation card.
    legend: {
      // plain (default) WRAPS to multiple rows so all 5 classes show at once;
      // "scroll" would paginate (1/3) and hide classes in a static screenshot.
      orient: "horizontal" as const,
      bottom: 0,
      left: "center" as const,
      width: "96%",
      itemWidth: 11,
      itemHeight: 8,
      itemGap: 10,
      textStyle: { fontFamily: SERIF, color: SLATE, fontSize: 10.5 },
      data: assetClasses,
    },
    series: [
      {
        name: "Target",
        type: "pie" as const,
        radius: ["26%", "42%"],
        center: ["50%", "44%"],
        label: { show: false },
        labelLine: { show: false },
        emphasis: { scale: true, scaleSize: 5 },
        data: innerData,
      },
      {
        name: "Current",
        type: "pie" as const,
        radius: ["46%", "62%"],
        center: ["50%", "44%"],
        label: { show: false },
        labelLine: { show: false },
        emphasis: { scale: true, scaleSize: 4 },
        data: outerData,
      },
    ],
    tooltip: {
      ...houseChartBase.tooltip,
      formatter: (params: { seriesName: string; name: string; value: number }) =>
        `<span style="font-family:${SERIF};font-size:13px"><b>${params.seriesName}</b><br/>${params.name}: <b>${params.value}%</b></span>`,
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
