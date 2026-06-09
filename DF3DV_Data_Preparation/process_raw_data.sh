#!/usr/bin/env bash

# heic to jpg
find ./  -type f -iname '*.HEIC' -exec mogrify -format jpg {} \; -exec rm {} \;
# to JPG
find ./ -type f   \( -iname '*.png' -o -iname '*.jpeg' -o -iname '*.jpg' \) | while read -r f; do   tmp="${f%.*}.TMP";   mv "$f" "$tmp";   mv "$tmp" "${f%.*}.JPG"; done
# check JPG
find . -type f -regex '.*\.\(jpg\|jpeg\|JPG\|JPEG\)' ! -name '*.JPG'
# auto-orient
find ./ -type f -iname '*.JPG' -execdir jhead -v -autorot {} +
# check JPG
find . -type f -regex '.*\.\(jpg\|jpeg\|JPG\|JPEG\)' ! -name '*.JPG'
# how many is oriented
exiftool -r -ext jpg -if '$Orientation and $Orientation ne 1' -FileName -Orientation -n ./

