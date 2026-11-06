`weather.csv` contains the first 240 hourly temperature measurements of 2014
from Song Chen (2015), *Beijing PM2.5*, UCI Machine Learning Repository,
https://doi.org/10.24432/C5JS49, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

I retained the original temperature values in degrees Celsius, combined the
calendar columns into `timestamp`, and renamed `TEMP` to `value`. The timestamps
remain the source's local station time; I did not convert them to UTC.
The source archive's checksum and extraction live in `src/sensor_forecast/data.py`.
This excerpt lets CI fit real models on public observations without a network download.
