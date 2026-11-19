#!/bin/sh
# Fetch MersenneTwister.dat and MersenneTwister.raw for the mt benchmarks
# from NVIDIA's official oclMersenneTwister sample, instead of HeCBench
# hosting NVIDIA's files. See DATA-NOTICE in this directory.
set -eu
cd "$(dirname "$0")"
[ -f MersenneTwister.dat ] && [ -f MersenneTwister.raw ] && exit 0

url=https://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclMersenneTwister.zip
md5=74ee39ff40b3ddc2fff22f8d5d27fbb8
zip=oclMersenneTwister.zip

curl -fsSL -o "$zip" "$url"
echo "$md5  $zip" | md5sum -c --quiet -
unzip -j -o -q "$zip" \
    '*/oclMersenneTwister/data/MersenneTwister.dat' \
    '*/oclMersenneTwister/data/MersenneTwister.raw'
rm -f "$zip"
