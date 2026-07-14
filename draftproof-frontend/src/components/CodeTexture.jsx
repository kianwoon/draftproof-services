// Decorative drifting "code rain" backdrop. Rendered as two horizontally
// scrolling tracks (row A drifts left, row B drifts right).
//
// PERF: the drift MUST be a CSS transform on the track *element* (an HTML
// <div>), NOT on an SVG <g>. Animating transform on an inner SVG group is not
// compositor-accelerated — Chrome runs a full main-thread style + LAYOUT recalc
// every frame (measured: ~240 layouts/sec on a high-refresh display, ×2 rows,
// for a ~0.2-opacity decoration). Animating an HTML element's transform runs
// entirely on the compositor thread. Each track holds two identical SVG copies
// side by side and translates by exactly one copy width (-50%), so the loop is
// seamless. See the landing-hero CPU trace, 2026-07-14.
//
// Only the lines that were ever visible are rendered: the previous SVG <mask>
// clipped everything below y=320 (of the 640 viewBox), so the lower four lines
// per row never painted. The horizontal edge-fade now lives on the container as
// a CSS mask-image (see .code-texture in 02-landing-hero.css).

const ROW_A_LINES = [
  { y: 64, text: '0101  source.check()  1100  citation.match  0010  grounded=true  1011' },
  { y: 136, text: 'claim.verify  1001  source:linked  0110  citation.add  1010  review.only' },
  { y: 208, text: '0011  evidence.fit  1110  no_verdict  0101  grounded=true  1001' },
  { y: 280, text: 'source.check()  0111  claim.supported  1010  citation.gap  0001' },
];

const ROW_B_LINES = [
  { y: 100, text: 'grounded=true  0110  citation.match  1011  claim.verify  0010' },
  { y: 172, text: '1100  source.integrity  0001  review.signal  1010  evidence.fit' },
  { y: 244, text: 'citation.gap  0101  source.check()  1110  no_ai_verdict  1001' },
  { y: 316, text: '0010  claim.supported  1011  source.map  0110  citation.fix' },
];

export default function CodeTexture({ className = '', id = 'codeTexture', preserveAspectRatio = 'none' }) {
  return (
    <div className={`code-texture ${className}`.trim()} aria-hidden="true" data-id={id}>
      <div className="code-texture-track code-row code-row-a">
        <CodeRowSvg lines={ROW_A_LINES} preserveAspectRatio={preserveAspectRatio} />
        <CodeRowSvg lines={ROW_A_LINES} preserveAspectRatio={preserveAspectRatio} />
      </div>
      <div className="code-texture-track code-row code-row-b">
        <CodeRowSvg lines={ROW_B_LINES} preserveAspectRatio={preserveAspectRatio} />
        <CodeRowSvg lines={ROW_B_LINES} preserveAspectRatio={preserveAspectRatio} />
      </div>
    </div>
  );
}

function CodeRowSvg({ lines, preserveAspectRatio }) {
  return (
    <svg
      className="code-texture-svg"
      viewBox="0 0 1440 640"
      focusable="false"
      preserveAspectRatio={preserveAspectRatio}
    >
      <g className="code-texture-layer">
        {lines.map((line) => (
          <CodeLine key={line.y} y={line.y} text={line.text} />
        ))}
      </g>
    </svg>
  );
}

function CodeLine({ y, text }) {
  return (
    <text x="0" y={y}>
      <tspan>{text}</tspan>
      <tspan dx="64">{text}</tspan>
      <tspan dx="64">{text}</tspan>
    </text>
  );
}
