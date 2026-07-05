export default function FixFirstChecklist({ items, onSelectParagraph, title, kicker, intro }) {
  if (!Array.isArray(items) || items.length === 0) return null;

  return (
    <section className="fix-first-card" aria-label={title}>
      <div className="fix-first-head">
        <span>{kicker}</span>
        <h2>{title}</h2>
      </div>
      {intro && <p className="fix-first-intro">{intro}</p>}
      <div className="fix-first-list">
        {items.map((item, index) => {
          const clickable = item.paragraphId && typeof onSelectParagraph === 'function';
          const content = (
            <>
              <span className="fix-first-index">{index + 1}</span>
              <span className="fix-first-copy">
                <strong>{item.title}</strong>
                {item.body && <em>{item.body}</em>}
              </span>
              {item.label && <span className="fix-first-label">{item.label}</span>}
            </>
          );
          if (clickable) {
            return (
              <button
                key={`${item.paragraphId}-${index}`}
                type="button"
                className="fix-first-item"
                onClick={() => onSelectParagraph(item.paragraphId)}
              >
                {content}
              </button>
            );
          }
          return (
            <div key={`${item.title}-${index}`} className="fix-first-item">
              {content}
            </div>
          );
        })}
      </div>
    </section>
  );
}
