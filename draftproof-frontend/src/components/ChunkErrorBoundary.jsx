import { Component } from 'react';

// Guards the lazy-loaded routes. After a rolling deploy the old hashed JS chunks are gone,
// so an already-open tab navigating to a code-split page (e.g. /report) can hit a rejected
// dynamic import(). Without a boundary that throw unmounts the whole app → blank page.
// Here we reload ONCE to pull the current build (guarded against a reload loop); if that
// still fails we render a readable fallback instead of a blank screen.
const CHUNK_ERROR_RE = /Loading chunk|dynamically imported module|Importing a module script failed|error loading dynamically imported module/i;
const RELOAD_FLAG = 'chunkReloadAttempted';

export default class ChunkErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error) {
    const isChunkError = CHUNK_ERROR_RE.test(error?.message || '');
    if (!isChunkError) return;
    try {
      if (!sessionStorage.getItem(RELOAD_FLAG)) {
        sessionStorage.setItem(RELOAD_FLAG, '1');
        window.location.reload();
      }
    } catch {
      window.location.reload();
    }
  }

  render() {
    if (this.state.failed) return this.props.fallback ?? null;
    return this.props.children;
  }
}
