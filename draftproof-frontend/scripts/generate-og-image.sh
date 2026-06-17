#!/usr/bin/env bash
# Regenerate the social share image (public/og-image.png, 1200x630).
#
# The DraftProof logo is reused verbatim by cropping it out of the existing
# image, so only the text below changes. Edit the four text lines below to
# update the card, then run:  bash scripts/generate-og-image.sh
#
# Requires ImageMagick (`magick`). After running, commit the new PNG. NOTE:
# social scrapers cache by URL — bump the `?v=` suffix on DEFAULT_IMAGE in
# src/seoMetadata.js so platforms re-fetch the updated image.
set -euo pipefail

cd "$(dirname "$0")/.."
PUB=public/og-image.png
TMP_LOGO=$(mktemp -t dp_og_logo).png
TMP_OUT=$(mktemp -t dp_og_out).png

WORDMARK='DraftProof'
TAGLINE='Keep the thinking in your draft yours'
PILLS='Citation gaps      ·      Weak claims      ·      AI-like signals'
SUBLINE='Built for students, educators, and researchers'

FONT_BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG='/System/Library/Fonts/Supplemental/Arial.ttf'

# Lift the logo (with its navy margin) out of the current image.
magick "$PUB" -crop 282x182+0+68 +repage "$TMP_LOGO"

magick -size 1200x630 xc:'#0D1B2A' \
  "$TMP_LOGO" -geometry +0+68 -composite \
  -font "$FONT_BOLD" -fill '#FFFFFF' -pointsize 94 -annotate +290+178 "$WORDMARK" \
  -font "$FONT_REG"  -fill '#D7DEE6' -pointsize 38 -annotate +292+232 "$TAGLINE" \
  -font "$FONT_REG"  -fill '#3BA876' -pointsize 32 -annotate +292+308 "$PILLS" \
  -font "$FONT_REG"  -fill '#8A97A6' -pointsize 27 -annotate +292+366 "$SUBLINE" \
  "$TMP_OUT"

cp "$TMP_OUT" "$PUB"
rm -f "$TMP_LOGO" "$TMP_OUT"
echo "Wrote $PUB"
