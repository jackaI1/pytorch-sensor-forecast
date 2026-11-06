# Forecast errors

Lower MAE (mean absolute error) is better.

| Model | Hours ahead | MAE | 95% bootstrap interval | Origins |
|---|---:|---:|---:|---:|
| seasonal_naive | 1 | 2.661 | 2.304 to 3.036 | 56 |
| seasonal_naive | 6 | 2.482 | 1.786 to 3.304 | 56 |
| seasonal_naive | 24 | 2.357 | 1.964 to 2.804 | 56 |
| gradient_boosting | 1 | 1.357 | 1.050 to 1.823 | 56 |
| gradient_boosting | 6 | 2.193 | 1.563 to 2.981 | 56 |
| gradient_boosting | 24 | 2.493 | 1.770 to 3.336 | 56 |
| lstm | 1 | 1.437 | 1.072 to 1.836 | 56 |
| lstm | 6 | 6.763 | 5.728 to 7.784 | 56 |
| lstm | 24 | 2.773 | 2.167 to 3.438 | 56 |

Descriptive uncertainty on this test period, not future forecast bounds
