#!/bin/sh
convert -density 150 -depth 10 -quality 95 DFTK_2to1.pdf DFTK_2000x1000.png
convert -density "7.5" -depth 10 -quality 95 DFTK_2to1.pdf DFTK_100x50.png
convert -density 60 -depth 10 -quality 95 DFTK_3to1.pdf DFTK_750x250.png
convert -density 12 -depth 10 -quality 95 DFTK_3to1.pdf DFTK_150x50.png
convert -density 300 -depth 10 -quality 95 DFTK.pdf DFTK.png
