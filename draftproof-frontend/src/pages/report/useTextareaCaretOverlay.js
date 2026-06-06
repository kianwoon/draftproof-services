import { useCallback, useEffect, useRef, useState } from 'react';

const CARET_MIRROR_PROPS = [
  'boxSizing',
  'width',
  'borderTopWidth',
  'borderRightWidth',
  'borderBottomWidth',
  'borderLeftWidth',
  'paddingTop',
  'paddingRight',
  'paddingBottom',
  'paddingLeft',
  'fontFamily',
  'fontSize',
  'fontWeight',
  'fontStyle',
  'letterSpacing',
  'lineHeight',
  'textTransform',
  'textIndent',
  'tabSize',
  'wordBreak',
];

function measureTextareaCaret(textarea) {
  if (!textarea || typeof document === 'undefined') return null;
  const style = window.getComputedStyle(textarea);
  const mirror = document.createElement('div');
  CARET_MIRROR_PROPS.forEach((prop) => {
    mirror.style[prop] = style[prop];
  });
  mirror.style.position = 'absolute';
  mirror.style.visibility = 'hidden';
  mirror.style.whiteSpace = 'pre-wrap';
  mirror.style.overflowWrap = 'break-word';
  mirror.style.left = '-9999px';
  mirror.style.top = '0';
  mirror.textContent = textarea.value.slice(0, textarea.selectionStart);

  const marker = document.createElement('span');
  marker.textContent = textarea.value.slice(textarea.selectionStart, textarea.selectionStart + 1) || '.';
  mirror.appendChild(marker);
  document.body.appendChild(mirror);
  const lineHeight = Number.parseFloat(style.lineHeight) || Number.parseFloat(style.fontSize) * 1.78;
  const position = { left: marker.offsetLeft, top: marker.offsetTop, height: lineHeight };
  mirror.remove();
  return position;
}

export default function useTextareaCaretOverlay(textareaRef) {
  const [caret, setCaret] = useState({ visible: false, left: 0, top: 0, height: 0 });
  const frameRef = useRef(null);

  const hideCaret = useCallback(() => {
    setCaret((current) => (current.visible ? { ...current, visible: false } : current));
  }, []);

  const updateCaret = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea || document.activeElement !== textarea || textarea.selectionStart !== textarea.selectionEnd) {
      hideCaret();
      return;
    }

    const position = measureTextareaCaret(textarea);
    if (!position) {
      hideCaret();
      return;
    }

    setCaret({
      visible: true,
      left: position.left - textarea.scrollLeft,
      top: position.top - textarea.scrollTop,
      height: position.height,
    });
  }, [hideCaret, textareaRef]);

  const scheduleCaretUpdate = useCallback(() => {
    if (frameRef.current) {
      window.cancelAnimationFrame(frameRef.current);
    }
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null;
      updateCaret();
    });
  }, [updateCaret]);

  useEffect(() => () => {
    if (frameRef.current) {
      window.cancelAnimationFrame(frameRef.current);
    }
  }, []);

  return { caret, hideCaret, scheduleCaretUpdate };
}
