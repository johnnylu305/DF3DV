#!/usr/bin/env bash


for dir in */; do
    dirname="${dir%/}"
    echo "==> Zipping $dirname into ${dirname}.zip"
    zip -vr "${dirname}.zip" "$dirname"
done
