/**
 * CitationSankey — ECharts sankey diagram: NBA claims → source doc_ids.
 * This is the "trust surface" — shows which documents back each recommendation.
 */
import ReactECharts from "echarts-for-react";
import { houseChartBase, CLAY, OLIVE, AMBER, SLATE, GRAY_300, GRAY_500, SERIF } from "./theme";

export interface SankeyLink {
  source: string;
  target: string;
  value?: number;
}

interface Props {
  links: SankeyLink[];
  height?: number;
}

// Deterministic colour assignment for source nodes (claims)
const CLAIM_COLORS = [CLAY, OLIVE, AMBER];

export default function CitationSankey({ links, height = 300 }: Props) {
  // Derive unique node list preserving insertion order
  const nodeSet = new Set<string>();
  links.forEach((l) => {
    nodeSet.add(l.source);
    nodeSet.add(l.target);
  });
  const nodes = Array.from(nodeSet);

  const sources = Array.from(new Set(links.map((l) => l.source)));

  const nodeData = nodes.map((name) => {
    const srcIdx = sources.indexOf(name);
    return {
      name,
      itemStyle: srcIdx >= 0
        ? { color: CLAIM_COLORS[srcIdx % CLAIM_COLORS.length] }
        : { color: GRAY_300, borderColor: GRAY_500 },
      label: {
        fontFamily: SERIF,
        fontSize: 12,
        color: SLATE,
      },
    };
  });

  const edgeData = links.map((l) => ({
    source: l.source,
    target: l.target,
    value: l.value ?? 1,
    lineStyle: {
      color: "gradient" as const,
      opacity: 0.45,
      curveness: 0.5,
    },
  }));

  const option = {
    ...houseChartBase,
    series: [
      {
        type: "sankey" as const,
        data: nodeData,
        links: edgeData,
        orient: "horizontal" as const,
        nodeWidth: 16,
        nodeGap: 14,
        layoutIterations: 32,
        label: {
          fontFamily: SERIF,
          fontSize: 12,
          color: SLATE,
        },
        emphasis: {
          focus: "adjacency" as const,
          lineStyle: { opacity: 0.8 },
        },
      },
    ],
    tooltip: {
      ...houseChartBase.tooltip,
      formatter: (params: {
        dataType?: string;
        name?: string;
        data?: { source?: string; target?: string; value?: number };
        value?: number;
      }) => {
        if (params.dataType === "edge") {
          return `<span style="font-family:${SERIF};font-size:13px"><b>${params.data?.source}</b><br/>→ <i>${params.data?.target}</i></span>`;
        }
        return `<span style="font-family:${SERIF};font-size:13px">${params.name}</span>`;
      },
    },
    // No grid — sankey fills the container
    backgroundColor: "transparent",
  };

  return (
    <ReactECharts
      option={option}
      style={{ width: "100%", height }}
      opts={{ renderer: "svg" }}
    />
  );
}
