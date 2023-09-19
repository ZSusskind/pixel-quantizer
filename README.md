![Example using PC-9800 palette (16 colors from 12-bit palette); original image is by user Argash [here](https://commons.wikimedia.org/wiki/File:Austin_Evening.jpg)](example/highlight_pc98.png)

Automatically quantize and pixelate an image

TODO: Write better README

Installation:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Example usage:
```
./convert_image.py <input> <output> --block_size 3 --palette_size 16 --accent_size 9 --denoise 1 --dither 4 --quantize 4 4 4 --resample
```
