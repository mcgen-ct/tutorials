##!/usr/bin/env bash

# Loop over the pdfs.
for PDF in *.pdf; do
    PNG=${PDF%.*}.png
    rm -f $PNG
    convert -density 300 $PDF -resize 400x $PNG
done
