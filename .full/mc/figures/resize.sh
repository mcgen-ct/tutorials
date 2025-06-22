##!/usr/bin/env bash

# Loop over the pdfs.
for PDF in *.pdf; do
    PNG=${PDF%.*}.png
    rm -f $PNG
    convert $PDF -resize 450x $PNG
done
