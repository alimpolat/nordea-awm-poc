/**
 * AllocationDonut — current allocation as a single, clearly-labelled donut,
 * paired with a drift table (current vs IPS target, colour-coded by breach size).
 *
 * Replaces the old two-concentric-ring design: same-hued inner/outer rings that
 * differed only by opacity were unreadable. One labelled ring carries the "where
 * is the money" story; the table carries the "vs target" story with exact numbers.
 */
import ReactECharts from "echarts-for-react";
import {
  houseChartBase,
  SERIF,
  MONO,
  CLAY,
  OLIVE,
  AMBER,
  RUST,
  GRAY_500,
  GRAY_300,
  SLATE,
  IVORY,
} from "./theme";

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

const PALETTE = [CLAY, OLIVE, AMBER, RUST, GRAY_500, "#5B8DB8"];
const colorFor = (name: string, idx: number) =>
  ASSET_COLORS[name] ?? PALETTE[idx % PALETTE.length];

/** drift magnitude → colour (mirrors the Drift Bars semantics) */
function driftColor(absPp: number): string {
  if (absPp > 4) return RUST;
  if (absPp >= 2) return AMBER;
  return OLIVE;
}

export default function AllocationDonut({ target, current, height = 360 }: Props) {
  // union of keys, ordered by current weight (largest first)
  const names = Array.from(
    new Set([...Object.keys(current), ...Object.keys(target)]),
  ).sort((a, b) => (current[b] ?? 0) - (current[a] ?? 0));

  const rows = names.map((name, i) => {
    const cur = (current[name] ?? 0) * 100;
    const tgt = (target[name] ?? 0) * 100;
    const drift = cur - tgt;
    return { name, cur, tgt, drift, color: colorFor(name, i) };
  });

  const donutHeight = Math.max(190, height - 150);

  const option = {
    ...houseChartBase,
    legend: { show: false },
    title: {
      text: "CURRENT",
      subtext: "vs IPS target",
      left: "center",
      top: "39%",
      textStyle: {
        fontFamily: MONO,
        fontSize: 10,
        color: GRAY_500,
        fontWeight: 600,
        letterSpacing: 1,
      },
      subtextStyle: { fontFamily: SERIF, fontSize: 11, color: SLATE },
      itemGap: 3,
    },
    series: [
      {
        name: "Current allocation",
        type: "pie" as const,
        radius: ["44%", "74%"],
        center: ["50%", "50%"],
        avoidLabelOverlap: true,
        itemStyle: {
          borderColor: IVORY,
          borderWidth: 2,
          borderRadius: 4,
        },
        label: {
          show: true,
          position: "outside" as const,
          formatter: "{b|{b}}\n{d|{c}%}",
          rich: {
            b: { fontFamily: SERIF, fontSize: 11, color: SLATE, lineHeight: 14 },
            d: { fontFamily: MONO, fontSize: 11.5, color: SLATE, fontWeight: 700 },
          },
        },
        labelLine: { show: true, length: 8, length2: 8, lineStyle: { color: GRAY_300 } },
        emphasis: {
          scale: true,
          scaleSize: 6,
          itemStyle: { shadowBlur: 12, shadowColor: "rgba(0,0,0,0.18)" },
        },
        data: rows.map((r) => ({
          name: r.name,
          value: +r.cur.toFixed(1),
          itemStyle: { color: r.color },
        })),
      },
    ],
    tooltip: {
      ...houseChartBase.tooltip,
      formatter: (p: { name: string; value: number }) => {
        const row = rows.find((r) => r.name === p.name);
        const d = row ? row.drift : 0;
        const sign = d > 0 ? "+" : "";
        const dc = row ? driftColor(Math.abs(d)) : GRAY_500;
        return `<span style="font-family:${SERIF};font-size:13px"><b>${p.name}</b><br/>Current: <b>${p.value}%</b> &nbsp;·&nbsp; Target: ${row?.tgt.toFixed(0)}%<br/>Drift: <b style="color:${dc}">${sign}${d.toFixed(1)} pp</b></span>`;
      },
    },
  };

  return (
    <div>
      <ReactECharts
        option={option}
        style={{ width: "100%", height: donutHeight }}
        opts={{ renderer: "svg" }}
      />

      {/* Drift table — exact current vs target with colour-coded breach */}
      <table className="w-full mt-1" style={{ fontFamily: SERIF }}>
        <thead>
          <tr style={{ color: GRAY_500, fontFamily: MONO }} className="text-[9px] uppercase tracking-wider">
            <th className="text-left font-normal pb-1">Asset class</th>
            <th className="text-right font-normal pb-1">Curr</th>
            <th className="text-right font-normal pb-1">Tgt</th>
            <th className="text-right font-normal pb-1 pr-1">Drift</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const sign = r.drift > 0 ? "+" : "";
            const onTarget = Math.abs(r.drift) < 0.05;
            return (
              <tr key={r.name} className="border-t" style={{ borderColor: "#EFEDE6" }}>
                <td className="py-[3px]">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block rounded-sm"
                      style={{ width: 9, height: 9, background: r.color }}
                    />
                    <span className="text-[11.5px]" style={{ color: SLATE }}>{r.name}</span>
                  </span>
                </td>
                <td className="text-right text-[11.5px] font-semibold" style={{ fontFamily: MONO, color: SLATE }}>
                  {r.cur.toFixed(1)}
                </td>
                <td className="text-right text-[11px]" style={{ fontFamily: MONO, color: GRAY_500 }}>
                  {r.tgt.toFixed(0)}
                </td>
                <td
                  className="text-right text-[11px] font-semibold pr-1"
                  style={{ fontFamily: MONO, color: onTarget ? GRAY_500 : driftColor(Math.abs(r.drift)) }}
                >
                  {onTarget ? "0.0" : `${sign}${r.drift.toFixed(1)}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[9px] mt-1.5" style={{ fontFamily: MONO, color: GRAY_500 }}>
        Drift pp vs IPS · <span style={{ color: RUST }}>■</span>&gt;4 &nbsp;
        <span style={{ color: AMBER }}>■</span>2–4 &nbsp;
        <span style={{ color: OLIVE }}>■</span>≤2
      </p>
    </div>
  );
}
