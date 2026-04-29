"""DraftProof pipeline runner.

Usage:
  python3 draftproof.py input.txt               # from file
  cat essay.txt | python3 draftproof.py -       # from stdin
  python3 draftproof.py "Paste text here"        # from CLI arg
  python3 draftproof.py input.txt -o report.md   # save as Markdown
  python3 draftproof.py input.txt --verbose      # full detail
  python3 draftproof.py input.txt --rewrite      # detect + rewrite + report
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import run_pipeline, print_pipeline, print_pipeline_md


def resolve_input(args):
    if args.text == "-":
        return sys.stdin.read()
    if args.text and os.path.isfile(args.text):
        with open(args.text, "r") as f:
            return f.read()
    if args.text:
        return args.text
    return None


def main():
    parser = argparse.ArgumentParser(description="DraftProof — pre-submission integrity check")
    parser.add_argument("text", nargs="?", default=None,
                        help="File path, '-' for stdin, or inline text")
    parser.add_argument("-o", "--output", default=None,
                        help="Save report as Markdown to file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Full detail with metadata")
    parser.add_argument("--rewrite", action="store_true",
                        help="Run rewrite pass after detection (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--rewrite-model", default="claude-sonnet-4-20250514",
                        help="Claude model for rewrites (default: claude-sonnet-4-20250514)")
    parser.add_argument("--rewrite-passes", type=int, default=3,
                        help="Max rewrite iterations (default: 3)")
    args = parser.parse_args()

    text = resolve_input(args)
    if text is None:
        parser.print_help()
        print("\nError: no input provided. Pass a file path, '-' for stdin, or inline text.")
        sys.exit(1)

    label = args.text if args.text else "stdin"

    # Resolve API key for rewrite
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if args.rewrite and not api_key:
        print("Error: --rewrite requires an API key. Use --api-key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    print("=" * 72)
    print("DRAFTPROOF — Pre-Submission Integrity Check")
    print(f"Input: {label} ({len(text.split())} words)")
    print("=" * 72)

    mode_label = "detect → rewrite → report" if args.rewrite else "detect → report"
    print(f"\nRunning pipeline ({mode_label})...")
    report = run_pipeline(
        text,
        do_rewrite=args.rewrite,
        rewrite_api_key=api_key,
        rewrite_model=args.rewrite_model,
        rewrite_max_passes=args.rewrite_passes,
    )

    if args.output:
        import io
        buf = io.StringIO()
        print_pipeline_md(report, verbose=args.verbose, file=buf)
        with open(args.output, "w") as f:
            f.write(buf.getvalue())
        print(f"Report saved to {args.output}")

    print_pipeline(report, verbose=args.verbose)


if __name__ == "__main__":
    main()
