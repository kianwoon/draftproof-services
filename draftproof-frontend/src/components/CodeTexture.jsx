export default function CodeTexture({ className = '', id = 'codeTexture' }) {
  const gradientId = `${id}Fade`;
  const maskId = `${id}Mask`;

  return (
    <svg
      className={`code-texture ${className}`}
      viewBox="0 0 1440 640"
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
          <CodeLine y="352" text="1010  source.grounded  0011  citation.trace  1101  evidence.linked" />
          <CodeLine y="424" text="review.layer  0110  claim.context  1001  source.visible  0010" />
          <CodeLine y="496" text="1100  authorship.signal  0101  phrasing.check  1110  fix.ready" />
          <CodeLine y="568" text="source.check()  1001  citation.match  0010  grounded=true  0111" />
        </g>
        <g className="code-row code-row-b">
          <CodeLine y="100" text="grounded=true  0110  citation.match  1011  claim.verify  0010" />
          <CodeLine y="172" text="1100  source.integrity  0001  review.signal  1010  evidence.fit" />
          <CodeLine y="244" text="citation.gap  0101  source.check()  1110  no_ai_verdict  1001" />
          <CodeLine y="316" text="0010  claim.supported  1011  source.map  0110  citation.fix" />
          <CodeLine y="388" text="review.only  1101  trace.source  0101  evidence.fit  1000" />
          <CodeLine y="460" text="source.integrity  0011  cite.linked  1010  grounded.report" />
          <CodeLine y="532" text="0111  verify.context  1100  citation.add  0010  source.check()" />
          <CodeLine y="604" text="authorship.signal  0101  no_verdict  1110  actionable=true" />
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
