import { Component } from 'react';

// Guards the routed page (e.g. /scan, /report). After a rolling deploy the old hashed JS
// chunks are gone, so an already-open tab navigating to a code-split page can hit a rejected
// dynamic import() — but ANY other render error under this boundary (a transient SSE/data
// glitch, a null-ref, etc.) would otherwise leave the app permanently stuck on the fallback,
// since React error boundaries don't recover on their own.
// So on the FIRST error of a session we reload once to try to recover, guarded by sessionStorage
// so a persistently-failing page can't loop forever — a second error in the same session just
// renders the readable fallback with a manual Reload button.
// If sessionStorage itself is unavailable (e.g. Safari private browsing), we have no way to
// remember "already tried" across a reload — so we deliberately do NOT reload in that case,
// to avoid an unguarded infinite reload loop, and just show the static fallback instead.
const RELOAD_FLAG = 'routeErrorReloadAttempted';
const LAST_ERROR_KEY = 'routeErrorLastMessage';

export default class ChunkErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error, info) {
    // Reloading wipes the console, so log first — this is the only evidence of the underlying
    // bug once we've auto-reloaded past it.
    console.error('ChunkErrorBoundary caught a render error:', error, info?.componentStack);

    try {
      if (sessionStorage.getItem(RELOAD_FLAG)) return; // already tried once this session
      sessionStorage.setItem(RELOAD_FLAG, '1');
      try {
        sessionStorage.setItem(LAST_ERROR_KEY, String(error?.message || error));
      } catch {
        // best-effort diagnostic only, not required for the reload guard
      }
      window.location.reload();
    } catch {
      // sessionStorage unavailable — see comment above; fall through to the static fallback.
    }
  }

  render() {
    if (this.state.failed) return this.props.fallback ?? null;
    return this.props.children;
  }
}
