for f in *.tif; do  echo "Converting $f"; convert "$f" -resize 22% "$(basename "$f" .tif).jpg"; done

rm *.tif

