export default function CodeTexture({ className = '', id = 'codeTexture' }) {
  const gradientId = `${id}Fade`;
  const maskId = `${id}Mask`;

  return (
    <svg
      className={`code-texture ${className}`}
      viewBox="0 0 1440 320"
      aria-hidden="true"
      focusable="false"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#0D1B2A" stopOpacity="0" />
          <stop offset="12%" stopColor="#0D1B2A" stopOpacity="0.95" />
          <stop offset="82%" stopColor="#0D1B2A" stopOpacity="0.74" />
          <stop offset="100%" stopColor="#0D1B2A" stopOpacity="0" />
        </linearGradient>
        <mask id={maskId}>
          <rect width="1440" height="320" fill={`url(#${gradientId})`} />
        </mask>
      </defs>

      <g mask={`url(#${maskId})`} className="code-texture-layer">
        <g className="code-row code-row-a">
          <CodeLine y="64" text="0101  source.check()  1100  citation.match  0010  grounded=true  1011" />
          <CodeLine y="136" text="claim.verify  1001  source:linked  0110  citation.add  1010  review.only" />
          <CodeLine y="208" text="0011  evidence.fit  1110  no_verdict  0101  grounded=true  1001" />
          <CodeLine y="280" text="source.check()  0111  claim.supported  1010  citation.gap  0001" />
        </g>
        <g className="code-row code-row-b">
          <CodeLine y="100" text="grounded=true  0110  citation.match  1011  claim.verify  0010" />
          <CodeLine y="172" text="1100  source.integrity  0001  review.signal  1010  evidence.fit" />
          <CodeLine y="244" text="citation.gap  0101  source.check()  1110  no_ai_verdict  1001" />
        </g>
      </g>
    </svg>
  );
}

function CodeLine({ y, text }) {
  return (
    <text x="0" y={y}>
      <tspan>{text}</tspan>
      <tspan dx="64">{text}</tspan>
    </text>
  );
}
