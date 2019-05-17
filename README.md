# traffic_sign
This is the repo aims at traffic sign detection and recognition competition from data fountain.

Many thanks to the develop team of [mmdetection](https://github.com/open-mmlab/mmdetection), which this repo based on.

## Installation

- Install pytorch 1.0 and torchvision following the official instructions.
- Clone the repository.

  `git clone https://github.com/tkianai/traffic_sign`

- Compile cuda extensions.

  ```sh
  cd traffic_sign
  pip install cython
  ./compile.sh
  python setup.py develop
  ```

There are refined versions for this jobs, keep waiting...
